# -*- coding: utf-8 -*-
"""
train_ocr_sr.py — Dạy lại OCR GP-LPR trên chính ẢNH ĐẦU RA CỦA MẠNG LÀM NÉT (Notebook vòng 2).

VÌ SAO CẦN?
  Ở vòng 1, OCR chỉ được dạy bằng ảnh HR (nét thật). Nhưng ảnh GAN sinh ra có "kiểu" khác (chữ hơi nhòe,
  nét bị vẽ lại) nên OCR bị lệch miền: ảnh trông sạch, OCR rất tự tin (~0,89) nhưng vẫn đọc sai.
  Ở đây ta cho OCR học thẳng từ ảnh GAN (kèm nhãn thật) để nó quen với kiểu ảnh này.

CÁCH LÀM:
  1. Với mỗi mạng làm nét trong --sr_ckpts, tạo ảnh SR cho vài frame ngẫu nhiên của MỌI track train.
  2. Trộn với ảnh HR (để OCR không quên cách đọc ảnh nét).
  3. Fine-tune OCR (mặc định mở khóa toàn bộ, vì lần này cần thích nghi cả phần nhìn ảnh).
  4. Mỗi epoch đọc thử ảnh SR của tập VAL (tất cả frame) của từng mạng; lưu epoch tốt nhất
     (theo Exact Match 1 frame trung bình trên các mạng).

Đầu ra:
  <out>                          : checkpoint OCR mới (định dạng repo)
  <out_dir>/ocr_sr_log.csv       : log từng epoch
  <out_dir>/ocr_sr_summary.json  : so sánh trước / sau cho từng mạng và từng nhóm (bằng chứng)
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lrlpr.train_funcs import ocr_funcs as T
from lrlpr.utils import api as C
from lrlpr.utils.config import apply_yaml_defaults
from lrlpr.utils_test import eval_lib as E


class LRSubsetDataset(Dataset):
    """Trả về K frame LR (đã tiền xử lý) của mỗi track theo danh sách frame cho trước — để sinh ảnh SR."""

    def __init__(self, rows, frame_idx):
        self.rows = rows
        self.frame_idx = frame_idx  # mảng (N, K) các số frame 1..5

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        fr = [C.to_chw_float(C.prep_lr_u8(C.read_rgb(C.frame_path(r, "lr", int(f))))) for f in self.frame_idx[i]]
        return torch.from_numpy(np.stack(fr)), i


@torch.no_grad()
def generate_subset(G, rows, frame_idx, device, workers):
    """Chạy mạng làm nét trên K frame đã chọn của mỗi track. Trả về uint8 (N, K, 32, 96, 3)."""
    n, k = frame_idx.shape
    out = np.zeros((n, k, C.SR_H, C.SR_W, 3), dtype=np.uint8)
    loader = DataLoader(LRSubsetDataset(rows, frame_idx), batch_size=64, shuffle=False, num_workers=workers,
                        worker_init_fn=C.worker_init_fn)
    G.eval()
    t0 = time.time()
    for j, (x, idx) in enumerate(loader):
        b = x.shape[0]
        sr = G(x.view(-1, 3, C.LR_H, C.LR_W).to(device))
        u8 = (sr.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out[idx.numpy()] = u8.reshape(b, k, C.SR_H, C.SR_W, 3)
        if j % 50 == 0:
            print(f"      sinh ảnh SR: {min((j + 1) * 64, n)}/{n} track — {time.time() - t0:.0f}s")
    return out


def eval_all(model, val_rows, val_frames, device):
    """Đọc ảnh SR của tập val bằng OCR hiện tại, cho từng mạng. Trả về {tên: bảng tóm tắt}."""
    res = {}
    for name, frames in val_frames.items():
        preds, confs = E.read_frames(model, frames, device)
        df = E.build_pred_df(val_rows, preds, confs)
        res[name] = E.summarize(df, name, "val")
    return res


def macro_em(res):
    """Trung bình Exact Match 1 frame (nhóm ALL) trên tất cả các mạng."""
    return float(np.mean([s[s["group"] == "ALL"]["em_single"].iloc[0] for s in res.values()]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--ocr_init", required=True, help="OCR vòng 1 (gplpr_ft.pth) để bắt đầu")
    ap.add_argument("--out", required=True, help="nơi lưu OCR mới")
    ap.add_argument("--sr_ckpts", nargs="+", required=True, help="danh sách TÊN=ĐƯỜNG_DẪN checkpoint mạng làm nét")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_encoder", type=int, default=0)
    ap.add_argument("--sr_frames_per_track", type=int, default=2)
    ap.add_argument("--hr_frames_per_track", type=int, default=1)
    ap.add_argument("--max_train_tracks", type=int, default=0, help="0 = dùng hết (chỉ để chạy thử)")
    ap.add_argument("--max_val_tracks", type=int, default=0, help="0 = dùng hết (chỉ để chạy thử)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    apply_yaml_defaults(ap)
    args = ap.parse_args()

    C.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    srs = []
    for item in args.sr_ckpts:
        name, path = item.split("=", 1)
        if not os.path.exists(path):
            print(f"⚠ bỏ qua {name}: không thấy {path}")
            continue
        srs.append((name, path))
    if not srs:
        raise SystemExit("Không có checkpoint mạng làm nét nào hợp lệ.")
    print(f"Mạng làm nét dùng để dạy OCR: {[n for n, _ in srs]} | thiết bị {device}")

    # --- 1. Danh sách track
    tr = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_train.csv")), args.data_root)
    va = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_val.csv")), args.data_root)
    if args.max_train_tracks:
        tr = tr[:args.max_train_tracks]
    if args.max_val_tracks:
        va = C.stratified_subset(va, args.max_val_tracks, args.seed)

    # --- 2. OCR khởi đầu
    model, info = C.load_gplpr(args.repo, args.ocr_init, device)

    # --- 3. Dữ liệu huấn luyện: ảnh HR + ảnh SR của từng mạng
    print("Dựng dữ liệu huấn luyện ...")
    t0 = time.time()
    xs, ys = [], []
    Xhr, yhr = T.build_cache(tr, args.hr_frames_per_track, args.workers, args.seed, "HR train")
    xs.append(Xhr.numpy())
    ys += list(yhr)
    rng = np.random.RandomState(args.seed)
    k = args.sr_frames_per_track
    val_frames = {}
    for name, path in srs:
        print(f"   — {name}")
        G, _ = C.load_generator(args.repo, path, device)
        idx_tr = np.stack([rng.permutation(C.N_FRAMES)[:k] + 1 for _ in range(len(tr))])  # frame 1..5
        sr_tr = generate_subset(G, tr, idx_tr, device, args.workers)
        xs.append(sr_tr.reshape(-1, C.SR_H, C.SR_W, 3))
        ys += [r["plate_text"] for r in tr for _ in range(k)]
        # ảnh SR của val (cả 5 frame) để đo
        val_frames[name], _ms = E.run_generator(G, va, device, workers=args.workers)
        del G
        if device.type == "cuda":
            torch.cuda.empty_cache()
    X = torch.from_numpy(np.concatenate(xs)).to(device)
    del xs
    targets = model.converter.encode_list(ys, K=C.PLATE_LEN).to(device)
    print(f"Tổng {len(ys):,} ảnh huấn luyện (HR {len(yhr):,} + SR {len(ys) - len(yhr):,}) — dựng trong {time.time() - t0:.0f}s")

    # --- 4. Đo TRƯỚC khi dạy lại
    before = eval_all(model, va, val_frames, device)
    Xva_hr, yva_hr = T.build_cache(va, 1, args.workers, args.seed, "HR val")
    gva = [C.group_of(r) for r in va]
    hr_before, _ = T.evaluate(model, Xva_hr, yva_hr, gva, device)
    for name, s in before.items():
        row = s[s["group"] == "ALL"].iloc[0]
        print(f"[TRƯỚC] {name}: Exact Match 1 frame {row['em_single'] * 100:.2f}% | gộp 5 frame {row['em_mvcp_5'] * 100:.2f}%")
    print(f"[TRƯỚC] ảnh HR: {hr_before * 100:.2f}%")

    # --- 5. Huấn luyện
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    log_path = os.path.join(out_dir, "ocr_sr_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "macro_em_1frame_val", "em_hr_val", "giay"])
    n = X.shape[0]
    best, best_ep, t_start = macro_em(before) - 1e-9, 0, time.time()   # chỉ lưu nếu tốt hơn mốc trước
    for ep in range(1, args.epochs + 1):
        t_ep = time.time()
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()
        perm = torch.randperm(n, device=device)
        losses = []
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            x = X[idx].permute(0, 3, 1, 2).float().div(255.0)
            logits = model(x)[1]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets[idx].reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        res = eval_all(model, va, val_frames, device)
        m = macro_em(res)
        hr_em, _ = T.evaluate(model, Xva_hr, yva_hr, gva, device)
        sec = time.time() - t_ep
        print(f"Epoch {ep}/{args.epochs} — loss {np.mean(losses):.4f} — EM 1 frame (TB các mạng) {m * 100:.2f}% — "
              f"ảnh HR {hr_em * 100:.2f}% — {sec:.0f}s")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(float(np.mean(losses)), 5), round(m, 5), round(hr_em, 5), round(sec, 1)])
        if m > best:
            best, best_ep = m, ep
            C.save_gplpr(model, args.out, extra={"epoch": ep, "macro_em_1frame_val": m})
            print("   ↳ lưu checkpoint tốt nhất")

    # --- 6. Tóm tắt trước / sau
    if best_ep == 0:
        print("⚠ Không epoch nào tốt hơn OCR ban đầu — lưu bản cuối cùng để có file, nhưng KHÔNG nên dùng.")
        C.save_gplpr(model, args.out, extra={"epoch": args.epochs, "ghi_chu": "khong_cai_thien"})
    final, _ = C.load_gplpr(args.repo, args.out, device, verbose=False)
    after = eval_all(final, va, val_frames, device)
    hr_after, _ = T.evaluate(final, Xva_hr, yva_hr, gva, device)
    summary = {"epoch_tot_nhat": best_ep, "mang_lam_net": [n_ for n_, _ in srs], "so_anh_train": int(n),
               "em_hr_val_truoc": hr_before, "em_hr_val_sau": hr_after,
               "truoc": {k_: v.to_dict(orient="records") for k_, v in before.items()},
               "sau": {k_: v.to_dict(orient="records") for k_, v in after.items()},
               "tong_giay": round(time.time() - t_start, 1), "tham_so": vars(args),
               "ocr_khoi_dau": info, "checkpoint_sha256": C.sha256_file(args.out)}
    C.write_json(os.path.join(out_dir, "ocr_sr_summary.json"), summary)
    print("\nKẾT QUẢ (val, Exact Match 1 frame): ")
    for name in before:
        b = before[name][before[name]["group"] == "ALL"]["em_single"].iloc[0]
        a = after[name][after[name]["group"] == "ALL"]["em_single"].iloc[0]
        print(f"  {name}: {b * 100:.2f}% → {a * 100:.2f}%")
    print(f"  ảnh HR: {hr_before * 100:.2f}% → {hr_after * 100:.2f}%")


if __name__ == "__main__":
    main()
