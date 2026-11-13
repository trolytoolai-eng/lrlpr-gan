# -*- coding: utf-8 -*-
"""
report_lib.py — Tổng hợp kết quả, vẽ biểu đồ, dựng ví dụ và xuất bằng chứng (Notebook 05).

Mọi con số đều được TÍNH LẠI từ các file dự đoán từng track (pred_*.csv), không gõ tay,
nên có thể kiểm tra lại bất cứ lúc nào.
"""

import os
import random
import time

import numpy as np
import pandas as pd

from lrlpr.utils import api as C  # noqa: E402
from lrlpr.utils_test import eval_lib as E  # noqa: E402


# Quy ước tên nhánh:
#   M0..M4   : vòng 1 (xem eval_lib)
#   M5a/b/c  : vòng 2 — LPSRGAN huấn luyện có LOSS OCR thật (mỗi chữ cái = một trọng số loss OCR)
#   tên có dấu '+' : vòng 2 — dùng ĐÚNG ảnh của nhánh gốc nhưng đọc bằng OCR đã dạy lại trên ảnh GAN
#                    (ví dụ 'M4+' = ảnh của M4 đọc bằng OCR mới)
#   M6a/b/c  : vòng 3 — mạng làm nét NHIỀU FRAME (chồng 5 frame LR vào mạng cùng lúc) + loss OCR
BRANCH_ORDER = ["M0", "M1", "M2", "M3", "M4", "M5a", "M5b", "M5c", "M6a", "M6b", "M6c",
                "M2+", "M3+", "M4+", "M5a+", "M5b+", "M5c+", "M6a+", "M6b+", "M6c+", "HR"]
BRANCH_LABEL = {
    "M0": "M0 · LR phóng to",
    "M1": "M1 · CV cổ điển",
    "M2": "M2 · LPSRGAN công bố (zero-shot)",
    "M3": "M3 · LPSRGAN fine-tune",
    "M4": "M4 · LPSRGAN fine-tune + tăng cường",
    "M5a": "M5a · LPSRGAN + loss OCR (nhẹ)",
    "M5b": "M5b · LPSRGAN + loss OCR (vừa)",
    "M5c": "M5c · LPSRGAN + loss OCR (mạnh)",
    "M6a": "M6a · LPSRGAN nhiều frame + loss OCR (a)",
    "M6b": "M6b · LPSRGAN nhiều frame + loss OCR (b)",
    "M6c": "M6c · LPSRGAN nhiều frame + loss OCR (c)",
    "M6a+": "M6a+ · M6a đọc bằng OCR đã dạy lại",
    "M6b+": "M6b+ · M6b đọc bằng OCR đã dạy lại",
    "M6c+": "M6c+ · M6c đọc bằng OCR đã dạy lại",
    "M2+": "M2+ · M2 đọc bằng OCR đã dạy lại",
    "M3+": "M3+ · M3 đọc bằng OCR đã dạy lại",
    "M4+": "M4+ · M4 đọc bằng OCR đã dạy lại",
    "M5a+": "M5a+ · M5a đọc bằng OCR đã dạy lại",
    "M5b+": "M5b+ · M5b đọc bằng OCR đã dạy lại",
    "M5c+": "M5c+ · M5c đọc bằng OCR đã dạy lại",
    "HR": "HR · ảnh nét thật (cận trên)",
}
FUSION_LABEL = {"single": "đọc từng frame (trung bình)", "mvcp_3": "bỏ phiếu ký tự, 3 frame",
                "mvcp_5": "bỏ phiếu ký tự, 5 frame", "hc_3": "chuỗi tin cậy nhất, 3 frame",
                "hc_5": "chuỗi tin cậy nhất, 5 frame", "mv_3": "chuỗi phổ biến nhất, 3 frame",
                "mv_5": "chuỗi phổ biến nhất, 5 frame"}
COLORS = {"M0": "#9aa5b1", "M1": "#c9a227", "M2": "#4f86c6", "M3": "#2a9d8f", "M4": "#e76f51", "HR": "#444444",
          "M5a": "#8e6bbf", "M5b": "#6a4c93", "M5c": "#46307a",
          "M6a": "#d4a017", "M6b": "#b8860b", "M6c": "#8a6508",
          "M6a+": "#ecd48a", "M6b+": "#dcb85a", "M6c+": "#c9a03a",
          "M2+": "#9dbde0", "M3+": "#8fd0c8", "M4+": "#f2b3a3", "M5a+": "#c5b3e0", "M5b+": "#a98fd0", "M5c+": "#8a6fc0"}


def _label(b):
    return BRANCH_LABEL.get(b, b)


def _color(b):
    return COLORS.get(b, "#888888")


def base_branch(b):
    """Nhánh có dấu '+' dùng chung ảnh với nhánh gốc: 'M4+' -> 'M4'."""
    return b[:-1] if b.endswith("+") else b


def _groups_in(summary, split):
    """Các nhóm có trong bảng tóm tắt của một tập: 4 nhóm chuẩn trước, rồi nhóm khác (vd 'Test/Mercosur')."""
    present = [g for g in summary[summary["split"] == split]["group"].unique() if g != "ALL"]
    return [g for g in C.GROUPS if g in present] + sorted(g for g in present if g not in C.GROUPS)


def ok_col(fusion):
    """Tên cột đúng/sai ứng với cách gộp. 'single' (không gộp) -> dùng frame đầu tiên khi chọn ví dụ."""
    return "ok_1" if fusion == "single" else f"ok_{fusion}"


def pred_col(fusion):
    """Tên cột chuỗi dự đoán ứng với cách gộp."""
    return "pred_1" if fusion == "single" else fusion


# ----------------------------------------------------------------------------
# 1. Tổng hợp số liệu từ file dự đoán
# ----------------------------------------------------------------------------
def available_branches(results_dir):
    """Các nhánh đã có file dự đoán, theo thứ tự chuẩn."""
    found = [b for b in BRANCH_ORDER
             if any(os.path.exists(os.path.join(results_dir, b, f"pred_{s}.csv")) for s in ("val", "test"))]
    return found


def gather_summary(results_dir, out_csv=None):
    """Đọc mọi pred_*.csv và tính lại bảng tóm tắt (nhánh × tập × nhóm). Lưu summary.csv nếu có out_csv."""
    parts = []
    for b in available_branches(results_dir):
        for s in ("val", "test"):
            p = os.path.join(results_dir, b, f"pred_{s}.csv")
            if os.path.exists(p):
                parts.append(E.summarize(E.read_pred_csv(p), b, s))
    summ = pd.concat(parts, ignore_index=True)
    if out_csv:
        summ.to_csv(out_csv, index=False)
    return summ


def confusion_pairs(results_dir, fusion, out_csv=None, split="test"):
    """Đếm các cặp ký tự hay bị nhầm (ký tự thật -> ký tự đọc ra) của từng nhánh, để bàn về 'ảo giác'."""
    recs = []
    for b in available_branches(results_dir):
        p = os.path.join(results_dir, b, f"pred_{split}.csv")
        if not os.path.exists(p):
            continue
        df = E.read_pred_csv(p)
        for gt, pr in zip(df["plate_text"], df[pred_col(fusion)]):
            if len(gt) == len(pr):
                for a, c in zip(gt, pr):
                    if a != c:
                        recs.append((b, a, c))
    cp = (pd.DataFrame(recs, columns=["branch", "ky_tu_that", "ky_tu_doc_ra"])
          .value_counts().reset_index(name="so_lan").sort_values(["branch", "so_lan"], ascending=[True, False]))
    if out_csv:
        cp.to_csv(out_csv, index=False)
    return cp


# ----------------------------------------------------------------------------
# 2. Biểu đồ
# ----------------------------------------------------------------------------
def plot_overview(summary, fusion, out_png=None):
    """Biểu đồ cột: Exact Match (nhóm ALL) của từng nhánh, tách val và test."""
    import matplotlib.pyplot as plt
    col = f"em_{fusion}"
    fig, ax = plt.subplots(figsize=(9, 4.5))
    branches = [b for b in BRANCH_ORDER if b in set(summary["branch"])]
    w = 0.38
    for j, split in enumerate(["val", "test"]):
        vals = []
        for b in branches:
            r = summary[(summary["branch"] == b) & (summary["split"] == split) & (summary["group"] == "ALL")]
            vals.append(float(r[col].iloc[0]) * 100 if len(r) else np.nan)
        xs = np.arange(len(branches)) + (j - 0.5) * w
        bars = ax.bar(xs, vals, w, color=[_color(b) for b in branches], alpha=1.0 if split == "test" else 0.55,
                      edgecolor="white", label="test" if split == "test" else "val (nhạt)")
        for x, v in zip(xs, vals):
            if not np.isnan(v):
                ax.text(x, v + 0.8, f"{v:.1f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(branches)))
    ax.set_xticklabels([b for b in branches])
    ax.set_ylabel("Exact Match (%) — đúng cả 7 ký tự")
    ax.set_title(f"So sánh các nhánh — cách đọc: {FUSION_LABEL.get(fusion, fusion)}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def plot_groups(summary, fusion, split="test", out_png=None):
    """Biểu đồ cột theo từng nhóm Scenario × layout (để thấy nhóm nào khó nhất)."""
    import matplotlib.pyplot as plt
    col = f"em_{fusion}"
    groups = _groups_in(summary, split)
    branches = [b for b in BRANCH_ORDER if b in set(summary[summary["split"] == split]["branch"])]
    fig, ax = plt.subplots(figsize=(max(10, 1.2 * len(groups) * len(branches) / 2), 4.5))
    w = 0.8 / max(1, len(branches))
    for j, b in enumerate(branches):
        vals = []
        for g in groups:
            r = summary[(summary["branch"] == b) & (summary["split"] == split) & (summary["group"] == g)]
            vals.append(float(r[col].iloc[0]) * 100 if len(r) else np.nan)
        ax.bar(np.arange(len(groups)) + j * w - 0.4 + w / 2, vals, w, color=_color(b), label=b, edgecolor="white")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Exact Match (%)")
    ax.set_title(f"Theo từng nhóm — tập {split} — {FUSION_LABEL.get(fusion, fusion)}")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=min(len(branches), 8), fontsize=8)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


def plot_training(logs, out_png=None):
    """Đường cong huấn luyện của các lần fine-tune. logs: dict {tên nhánh: đường dẫn train_log.csv}."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for b, path in logs.items():
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        axes[0].plot(d["epoch"], d["train_loss"], marker="o", color=_color(b), label=b)
        axes[1].plot(d["epoch"], d["val_exact_match_1frame"] * 100, marker="o", color=_color(b), label=b)
    axes[0].set_title("Loss huấn luyện (L1 + VGG)")
    axes[1].set_title("Exact Match 1 frame trên val (%)")
    for a in axes:
        a.set_xlabel("epoch")
        a.spines[["top", "right"]].set_visible(False)
        a.legend(frameon=False)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=150)
    return fig


# ----------------------------------------------------------------------------
# 3. Tra cứu từng track và dựng ví dụ
# ----------------------------------------------------------------------------
class ResultStore:
    """Đọc và nhớ (cache) kết quả đã lưu để tra cứu nhanh, không cần GPU."""

    def __init__(self, results_dir):
        self.dir = results_dir
        self._pred, self._frames = {}, {}

    def pred(self, branch, split):
        key = (branch, split)
        if key not in self._pred:
            p = os.path.join(self.dir, branch, f"pred_{split}.csv")
            self._pred[key] = E.read_pred_csv(p).set_index("track") if os.path.exists(p) else None
        return self._pred[key]

    def frames(self, branch, split):
        key = (branch, split)
        if key not in self._frames:
            p = os.path.join(self.dir, base_branch(branch), f"sr_outputs_{split}.npz")  # 'M4+' dùng ảnh của 'M4'
            if os.path.exists(p):
                tracks, arr = E.load_frames_npz(p)
                self._frames[key] = ({t: i for i, t in enumerate(tracks)}, arr)
            else:
                self._frames[key] = None
        return self._frames[key]

    def branches(self, split):
        return [b for b in BRANCH_ORDER if self.pred(b, split) is not None]


def default_branches(store, split, best=None):
    """Các nhánh hiển thị mặc định (không quá nhiều cột): mốc, GAN công bố, M4, mô hình tốt nhất (+ bản OCR mới), HR."""
    have = store.branches(split)
    want = ["M0", "M1", "M2", "M4"]
    if best:
        want += [best, best + "+" if not best.endswith("+") else best]
    want += ["HR"]
    seen, out = set(), []
    for b in want:
        if b in have and b not in seen:
            out.append(b)
            seen.add(b)
    return out or have


def show_track(store, track, split, fusion, branches=None, out_png=None, show=True, best=None):
    """
    Hiện kết quả của MỘT track ở các nhánh cạnh nhau: 5 ảnh (frame 1..5) + chuỗi đọc được sau khi gộp.
    Đây là hàm để 'bật một ví dụ lên' khi thầy hỏi. best = mã nhánh tốt nhất (để luôn có mặt trong hình).
    """
    import matplotlib.pyplot as plt
    branches = branches or default_branches(store, split, best)
    first = store.pred(branches[0], split)
    gt = first.loc[track, "plate_text"]
    grp = first.loc[track, "group"]
    fig, axes = plt.subplots(C.N_FRAMES, len(branches), figsize=(2.6 * len(branches), 1.0 * C.N_FRAMES + 1.6),
                             squeeze=False)
    for j, b in enumerate(branches):
        df, fr = store.pred(b, split), store.frames(b, split)
        row = df.loc[track]
        fused, good = row[pred_col(fusion)], bool(row[ok_col(fusion)])
        for i in range(C.N_FRAMES):
            ax = axes[i, j]
            ax.axis("off")
            if fr is not None:
                ax.imshow(fr[1][fr[0][track], i], interpolation="nearest")
            ax.text(0.5, -0.12, str(row[f"pred_{i + 1}"]), transform=ax.transAxes, ha="center", va="top", fontsize=7,
                    color="#2a9d8f" if row[f"ok_{i + 1}"] else "#c0392b")
        axes[0, j].set_title(f"{_label(b).split(' · ')[0]}\n{fused} {'✓' if good else '✗'}", fontsize=10,
                             color="#2a9d8f" if good else "#c0392b")
    fig.suptitle(f"{track} · {grp} · nhãn thật: {gt} · cách đọc: {FUSION_LABEL.get(fusion, fusion)}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out_png:
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def pick_examples(store, split, fusion, best_branch, seed=42, per_cat=1):
    """
    Chọn các ca tiêu biểu (cố định theo seed) cho từng loại:
      gan_dung_cv_sai : mô hình tốt nhất đọc đúng, còn M0 và M1 đọc sai
      ao_giac         : mô hình tốt nhất đọc SAI nhưng rất tự tin (độ tin cậy trung bình ≥ 0.9)
      dem / ngay      : ca ban đêm (Scenario-B/Mercosur) / ban ngày (Scenario-A) đọc đúng
      brazilian / mercosur : đọc đúng theo từng kiểu biển
      sai_het         : không nhánh nào (trừ HR) đọc đúng
    """
    rng = random.Random(seed)
    oc = ok_col(fusion)
    best = store.pred(best_branch, split)
    others = [b for b in store.branches(split) if b != "HR"]
    all_ok = pd.DataFrame({b: store.pred(b, split)[oc] for b in others})
    conf = best[[f"conf_{i}" for i in range(1, C.N_FRAMES + 1)]].mean(axis=1)
    cats = {
        "gan_dung_cv_sai": best.index[(best[oc] == 1) & (all_ok.get("M0", 0) == 0) & (all_ok.get("M1", 0) == 0)],
        "ao_giac": best.index[(best[oc] == 0) & (conf >= 0.9)],
        "dem": best.index[(best[oc] == 1) & (best["group"] == "Scenario-B/Mercosur")],
        "ngay": best.index[(best[oc] == 1) & (best["scenario"] == "Scenario-A")],
        "brazilian": best.index[(best[oc] == 1) & (best["layout"] == "Brazilian")],
        "mercosur": best.index[(best[oc] == 1) & (best["layout"] == "Mercosur")],
        "sai_het": all_ok.index[(all_ok.sum(axis=1) == 0)],
    }
    out = {}
    for name, idx in cats.items():
        idx = sorted(list(idx))
        out[name] = rng.sample(idx, min(per_cat, len(idx))) if idx else []
    return out


def make_examples(store, split, fusion, best_branch, out_dir, seed=42, per_cat=1):
    """Dựng sẵn ảnh các ca tiêu biểu vào out_dir. Trả về danh sách (loại, track, đường dẫn ảnh)."""
    os.makedirs(out_dir, exist_ok=True)
    made = []
    for cat, tracks in pick_examples(store, split, fusion, best_branch, seed, per_cat).items():
        for t in tracks:
            safe = t.replace("/", "_").replace("\\", "_")  # mã track có dấu '/' nên phải đổi trước khi làm tên file
            path = os.path.join(out_dir, f"{split}_{cat}_{safe}.png")
            show_track(store, t, split, fusion, out_png=path, show=False, best=best_branch)
            made.append((cat, t, path))
    return made


# ----------------------------------------------------------------------------
# 4. Báo cáo dạng chữ + bằng chứng
# ----------------------------------------------------------------------------
def write_comparison_md(path, summary, chosen, timing_csv=None, ms_info=None):
    """Viết comparison.md: bảng so sánh các nhánh trên val và test + lựa chọn đã chốt trên val."""
    fusion = chosen["cach_gop_tot_nhat"]
    col = f"em_{fusion}"
    L = ["# So sánh các nhánh — LRLPR-26", "",
         f"- Cách đọc kết quả (chốt trên **val**): **{FUSION_LABEL.get(fusion, fusion)}** (`{fusion}`)",
         f"- Mô hình tốt nhất trên **val**: **{chosen['mo_hinh_tot_nhat']}** "
         f"({chosen['exact_match_val_cua_mo_hinh_tot_nhat'] * 100:.2f}%)",
         f"- Chốt lúc: {chosen['thoi_diem_chot']}", "", "## Exact Match toàn bộ (đúng cả 7 ký tự)", "",
         "| Nhánh | val | test | n(val) | n(test) |", "|---|---:|---:|---:|---:|"]
    for b in BRANCH_ORDER:
        sub = summary[(summary["branch"] == b) & (summary["group"] == "ALL")]
        if sub.empty:
            continue
        cells = {}
        for s in ("val", "test"):
            r = sub[sub["split"] == s]
            cells[s] = (f"{r[col].iloc[0] * 100:.2f}%", int(r["n"].iloc[0])) if len(r) else ("—", "—")
        L.append(f"| {_label(b)} | {cells['val'][0]} | {cells['test'][0]} | {cells['val'][1]} | {cells['test'][1]} |")
    for split_name, title in (("val", "Theo từng nhóm (tập val)"), ("test", "Theo từng nhóm (tập test)")):
        groups = _groups_in(summary, split_name)
        if not groups:
            continue
        L += ["", f"## {title}", "", "| Nhánh | " + " | ".join(groups) + " |", "|---|" + "---:|" * len(groups)]
        for b in BRANCH_ORDER:
            row = []
            for g in groups:
                r = summary[(summary["branch"] == b) & (summary["split"] == split_name) & (summary["group"] == g)]
                row.append(f"{r[col].iloc[0] * 100:.1f}%" if len(r) else "—")
            if any(x != "—" for x in row):
                L.append(f"| {_label(b)} | " + " | ".join(row) + " |")
    L += ["", "## Mọi cách đọc (nhóm ALL, tập test)", "", "| Nhánh | " + " | ".join(E.FUSION_COLS) + " |",
          "|---|" + "---:|" * len(E.FUSION_COLS)]
    for b in BRANCH_ORDER:
        r = summary[(summary["branch"] == b) & (summary["split"] == "test") & (summary["group"] == "ALL")]
        if len(r):
            L.append(f"| {_label(b)} | " + " | ".join(f"{r[f'em_{c}'].iloc[0] * 100:.1f}%" for c in E.FUSION_COLS) + " |")
    if ms_info:
        L += ["", "## Tốc độ mạng làm nét", "", "| Nhánh | ms / ảnh (GPU) |", "|---|---:|"]
        for b, ms in ms_info.items():
            L.append(f"| {_label(b)} | {ms:.3f} |")
    if timing_csv and os.path.exists(timing_csv):
        t = pd.read_csv(timing_csv)
        L += ["", "## Thời gian chạy từng bước", "", "| Bước | Giây | Trạng thái |", "|---|---:|---|"]
        for _, r in t.iterrows():
            L.append(f"| {r['step']} | {r['seconds']:.0f} | {r['status']} |")
        L.append(f"| **Tổng** | **{t['seconds'].sum():.0f}** (≈ {t['seconds'].sum() / 60:.1f} phút) | |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def write_manifest(roots, out_path, skip_ext=(".tmp",)):
    """
    Lập MANIFEST.json: danh sách mọi file kết quả / bằng chứng kèm kích thước, giờ sửa và sha256.
    roots: dict {tên: thư mục}. Dùng để chứng minh file không bị sửa sau khi chạy.
    """
    files = []
    for name, root in roots.items():
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, fnames in os.walk(root):
            for fn in sorted(fnames):
                if fn.endswith(skip_ext) or fn == os.path.basename(out_path):
                    continue
                p = os.path.join(dirpath, fn)
                files.append({"nhom": name, "file": os.path.relpath(p, root).replace("\\", "/"),
                              "bytes": os.path.getsize(p),
                              "sua_luc": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p))),
                              "sha256": C.sha256_file(p)})
    C.write_json(out_path, {"tao_luc": C.now_iso(), "so_file": len(files), "files": files})
    return len(files)


def write_evidence_readme(path):
    """Viết evidence/README.md: giải thích từng file bằng chứng và cách tái lập."""
    text = """# Bằng chứng thực nghiệm — LRLPR-26 (IT4343E)

Thư mục này gom mọi thứ cần để kiểm tra lại kết quả.

| File / thư mục | Nội dung |
|---|---|
| `environment_*.json` | GPU, phiên bản Python/PyTorch/thư viện của từng notebook |
| `timing.csv` | Giờ bắt đầu, kết thúc, số giây của từng bước (tổng thời gian chạy) |
| `patches.log` | Các bản vá đã áp lên repo LPSRGAN (commit 68fc78a) |
| `test_run.lock` | Giờ chạy tập test DUY NHẤT + lựa chọn đã chốt trên val trước đó |
| `MANIFEST.json` | Danh sách mọi file kết quả kèm sha256 |
| `logs/` | Toàn bộ output in ra của từng notebook (nếu đã lưu) |
| `notebooks/` | Notebook đã chạy kèm output (tải từ Colab) |

## Kết quả
`results/summary.csv` được tính lại từ `results/<nhánh>/pred_<tập>.csv` (mỗi track một dòng).
`results/<nhánh>/sr_outputs_<tập>.npz` chứa ảnh sau làm nét của mọi track; OCR luôn đọc từ chính các ảnh này.

## Tái lập
Chạy lần lượt notebook 00 → 05 trên Colab (thông số ở ô cấu hình đầu mỗi notebook).
Seed mặc định 42. Mọi tham số huấn luyện nằm trong `results/../ckpt/<lần chạy>/run_config.json`.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
