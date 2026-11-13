"""
TÁCH DỮ LIỆU LRLPR-26 -> TRAIN / VAL / TEST
Dự án: Nhận diện biển số xe độ phân giải thấp (IT4343E)

Nguyên tắc:
  1. Chia theo TRACK (5 frame của một xe luôn nằm cùng một tập).
  2. Chia phân tầng (stratified) theo (scenario x layout) -> val giữ đúng tỉ lệ 4 nhóm.
  3. Gom theo plate_text: các track cùng một biển số luôn nằm cùng một tập (tránh rò rỉ).
  4. Test = tập test chính thức, KHÔNG đụng tới khi huấn luyện / chỉnh tham số.
  5. Cố định seed, lưu danh sách ra file (kèm sha256) để tái lập.

Loại khỏi train/val (ghi vào excluded.csv, có lý do):
  - incomplete_files : track thiếu file (cần đủ 5 lr + 5 hr + annotations.json)
  - bad_label_format : nhãn không khớp định dạng layout (Brazilian LLLNNNN / Mercosur LLLNLNN)
  - plate_in_test    : biển số của train trùng với biển số trong test (tránh rò rỉ sang test)
Tập test chính thức KHÔNG bị loại gì; chỉ cảnh báo trong summary nếu có bất thường.

Cách chạy:
  python make_split.py --dataset <thư mục chứa train/ và test/> --out <thư mục lưu split>
"""

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

LAYOUT_RE = {
    "Brazilian": re.compile(r"^[A-Z]{3}[0-9]{4}$"),
    "Mercosur": re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"),
}
N_FRAMES = 5


def list_tracks(dataset_root, split_dir):
    """Trả về list (rel_path, scenario, layout) cho từng track."""
    base = os.path.join(dataset_root, split_dir)
    out = []
    if split_dir == "train":
        for scen in sorted(os.listdir(base)):
            for lay in sorted(os.listdir(os.path.join(base, scen))):
                d = os.path.join(base, scen, lay)
                for e in os.scandir(d):
                    if e.is_dir() and e.name.startswith("track_"):
                        out.append((f"{split_dir}/{scen}/{lay}/{e.name}", scen, lay))
    else:
        for e in os.scandir(base):
            if e.is_dir() and e.name.startswith("track_"):
                out.append((f"{split_dir}/{e.name}", "Test", "?"))
    return sorted(out)


def inspect_track(dataset_root, rel):
    """Đọc nhãn + đếm file của một track. Không ném lỗi: trả về cờ 'error'."""
    d = os.path.join(dataset_root, rel)
    n_lr = n_hr = 0
    has_ann = False
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name == "annotations.json":
                    has_ann = True
                elif e.name.startswith("lr-"):
                    n_lr += 1
                elif e.name.startswith("hr-"):
                    n_hr += 1
        text, layout, error = "", "", ""
        if has_ann:
            with open(os.path.join(d, "annotations.json"), "r", encoding="utf-8") as f:
                a = json.load(f)
            text, layout = a.get("plate_text", ""), a.get("plate_layout", "")
        else:
            error = "no_annotations"
    except Exception as ex:  # noqa: BLE001 - muốn bắt mọi lỗi I/O/JSON và ghi lại
        text, layout, error = "", "", f"read_error:{type(ex).__name__}"
    return {"plate_text": text, "layout": layout, "n_lr": n_lr, "n_hr": n_hr, "error": error}


def load_all(dataset_root, tracks, workers=32):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        infos = list(ex.map(lambda t: inspect_track(dataset_root, t[0]), tracks))
    rows = []
    for (rel, scen, lay), info in zip(tracks, infos):
        rows.append({"track": rel, "scenario": scen, "layout": info["layout"] or lay,
                     "plate_text": info["plate_text"], "n_lr": info["n_lr"],
                     "n_hr": info["n_hr"], "error": info["error"]})
    return rows


def is_complete(r, need_hr):
    if r["error"]:
        return False
    if r["n_lr"] != N_FRAMES:
        return False
    return (r["n_hr"] == N_FRAMES) if need_hr else True


def bad_label(r):
    pat = LAYOUT_RE.get(r["layout"])
    return pat is None or not pat.match(r["plate_text"])


def stratified_group_split(rows, val_ratio, seed):
    """Chia train/val: phân tầng theo (scenario, layout), gom theo plate_text."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for r in rows:
        groups[r["plate_text"]].append(r)
    strata = defaultdict(list)
    for text, items in groups.items():
        key = (items[0]["scenario"], items[0]["layout"])
        strata[key].append(text)
    stratum_tracks = Counter((r["scenario"], r["layout"]) for r in rows)

    val_texts = set()
    for key in sorted(strata):
        texts = sorted(strata[key])
        rng.shuffle(texts)
        target = round(stratum_tracks[key] * val_ratio)
        count = 0
        for t in texts:
            if count >= target:
                break
            val_texts.add(t)
            count += len(groups[t])
    train, val = [], []
    for r in rows:
        (val if r["plate_text"] in val_texts else train).append(r)
    return train, val


def write_csv(path, rows, split):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "scenario", "layout", "plate_text", "track"])
        for r in sorted(rows, key=lambda x: x["track"]):
            w.writerow([split, r["scenario"], r["layout"], r["plate_text"], r["track"]])


def write_excluded(path, items):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reason", "scenario", "layout", "plate_text", "track", "n_lr", "n_hr", "error"])
        for reason, r in items:
            w.writerow([reason, r["scenario"], r["layout"], r["plate_text"], r["track"],
                        r["n_lr"], r["n_hr"], r["error"]])


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dist(rows):
    return dict(sorted(Counter(f'{r["scenario"]}/{r["layout"]}' for r in rows).items()))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="thư mục chứa train/ và test/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("[1/6] Quét track train/test và kiểm tra file ...")
    train_all = load_all(args.dataset, list_tracks(args.dataset, "train"))
    test_all = load_all(args.dataset, list_tracks(args.dataset, "test"))
    print(f"      train={len(train_all):,}  test={len(test_all):,}")

    # --- cảnh báo với test (không loại gì) ---
    test_incomplete = [r for r in test_all if not is_complete(r, need_hr=False)]
    test_bad = [r for r in test_all if not r["error"] and bad_label(r)]
    print(f"[2/6] Test: thiếu file={len(test_incomplete)}, nhãn sai định dạng={len(test_bad)}")

    # --- loại khỏi train/val ---
    excluded = []
    keep = []
    for r in train_all:
        if not is_complete(r, need_hr=True):
            excluded.append(("incomplete_files", r))
        elif bad_label(r):
            excluded.append(("bad_label_format", r))
        else:
            keep.append(r)
    test_texts = {r["plate_text"] for r in test_all if r["plate_text"]}
    keep2 = []
    for r in keep:
        if r["plate_text"] in test_texts:
            excluded.append(("plate_in_test", r))
        else:
            keep2.append(r)
    reason_count = dict(Counter(reason for reason, _ in excluded))
    print(f"[3/6] Loại khỏi train/val: {len(excluded)} track {reason_count}")

    # --- thống kê biển số lặp ---
    tr_counts = Counter(r["plate_text"] for r in keep2)
    dup_plates = sum(1 for c in tr_counts.values() if c > 1)
    tracks_in_dup = sum(c for c in tr_counts.values() if c > 1)
    print(f"[4/6] Biển số lặp trong train: {dup_plates} biển ({tracks_in_dup} track) -> chia theo biển số")

    # --- chia ---
    print("[5/6] Chia train/val ...")
    train, val = stratified_group_split(keep2, args.val_ratio, args.seed)
    assert not ({r["plate_text"] for r in train} & {r["plate_text"] for r in val}), "Rò rỉ plate_text train/val!"
    assert not ({r["track"] for r in train} & {r["track"] for r in val}), "Trùng track train/val!"
    assert not ({r["plate_text"] for r in train + val} & test_texts), "Rò rỉ plate_text sang test!"

    paths = {
        "split_train.csv": (train, "train"),
        "split_val.csv": (val, "val"),
        "split_test.csv": (test_all, "test"),
    }
    for name, (rows, tag) in paths.items():
        write_csv(os.path.join(args.out, name), rows, tag)
    write_excluded(os.path.join(args.out, "excluded.csv"), excluded)

    summary = {
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_train": len(train), "n_val": len(val), "n_test": len(test_all),
        "dist_train": dist(train), "dist_val": dist(val),
        "dist_test_by_layout": dict(Counter(r["layout"] for r in test_all)),
        "excluded_from_train_val": reason_count,
        "duplicate_plate_texts_in_train": dup_plates,
        "tracks_with_duplicate_plate": tracks_in_dup,
        "test_warnings": {"incomplete": len(test_incomplete), "bad_label_format": len(test_bad)},
        "python": platform.python_version(),
        "sha256": {name: sha256(os.path.join(args.out, name)) for name in paths},
    }
    with open(os.path.join(args.out, "split_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[6/6] Xong. Tóm tắt:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
