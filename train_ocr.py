# -*- coding: utf-8 -*-
"""
train_ocr.py — Fine-tune mô hình OCR GP-LPR trên ảnh HR của tập train (Notebook 02).

Ý tưởng:
  * Bắt đầu từ trọng số GP-LPR đã huấn luyện sẵn (Rodosol.pth).
  * Dạy lại bằng ảnh HR của LRLPR-26 (đã CLAHE + đệm + 32x96, giống ảnh đích của mạng làm nét).
  * Mặc định ĐÓNG BĂNG phần encoder (đúng chỉ dẫn của thầy: đóng băng lớp đầu, chỉ chỉnh lớp sau).
  * Sau khi huấn luyện, OCR được ĐÓNG BĂNG và dùng chung cho mọi nhánh so sánh (M0..M4).

Để huấn luyện nhanh: ảnh HR được tiền xử lý MỘT LẦN rồi giữ trong bộ nhớ (cache), các epoch sau
chỉ chạy trên GPU. Cache gồm vài frame ngẫu nhiên cho mỗi track (--frames_per_track).

Đầu ra:
  * <out>                       : checkpoint tốt nhất (định dạng repo)
  * <out_dir>/ocr_train_log.csv : log từng epoch
  * <out_dir>/ocr_summary.json  : tóm tắt trước / sau fine-tune (bằng chứng)
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from lrlpr.train_funcs.ocr_funcs import build_cache, evaluate
from lrlpr.utils import api as C
from lrlpr.utils.config import apply_yaml_defaults


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, help="thư mục repo LPSRGAN đã vá")
    ap.add_argument("--data_root", required=True, help="thư mục chứa train/ và test/")
    ap.add_argument("--split_dir", required=True, help="thư mục chứa split_train.csv, split_val.csv")
    ap.add_argument("--init", required=True, help="checkpoint GP-LPR để bắt đầu (Rodosol.pth)")
    ap.add_argument("--out", required=True, help="nơi lưu checkpoint tốt nhất (gplpr_ft.pth)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--freeze_encoder", type=int, default=1)
    ap.add_argument("--frames_per_track", type=int, default=2)
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
    print(f"Thiết bị: {device} | seed {args.seed}")

    # --- 1. Đọc danh sách track
    tr = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_train.csv")), args.data_root)
    va = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_val.csv")), args.data_root)
    if args.max_train_tracks:
        tr = tr[:args.max_train_tracks]
    if args.max_val_tracks:
        va = va[:args.max_val_tracks]
    print(f"Số track: train {len(tr):,} | val {len(va):,}")

    # --- 2. Nạp OCR từ checkpoint khởi đầu
    model, info = C.load_gplpr(args.repo, args.init, device)

    # --- 3. Dựng cache ảnh HR (làm một lần)
    print("Dựng cache ảnh HR ...")
    Xtr, ytr = build_cache(tr, args.frames_per_track, args.workers, args.seed, "train")
    Xva, yva = build_cache(va, 1, args.workers, args.seed, "val")
    gva = [C.group_of(r) for r in va]
    Xtr_dev = Xtr.to(device)

    # --- 4. Đo OCR TRƯỚC khi fine-tune (để so sánh)
    em_before, grp_before = evaluate(model, Xva, yva, gva, device)
    print(f"Exact Match trên ảnh HR (val) TRƯỚC fine-tune: {em_before * 100:.2f}%")

    # --- 5. Đóng băng encoder (nếu bật)
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Tham số: {total:,} | đóng băng {frozen:,} ({frozen / total * 100:.1f}%)")
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # --- 6. Vòng huấn luyện
    log_path = os.path.join(out_dir, "ocr_train_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_exact_match", "seconds"])
    n = len(ytr)
    best_em, best_epoch, t_start = -1.0, 0, time.time()
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        if args.freeze_encoder:
            model.encoder.eval()  # giữ nguyên thống kê BatchNorm của phần đã đóng băng
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, args.batch):
            idx = perm[s:s + args.batch]
            x = Xtr_dev[idx.to(device)].permute(0, 3, 1, 2).float().div(255.0)
            target = model.converter.encode_list([ytr[i] for i in idx.tolist()], K=C.PLATE_LEN).to(device)
            logits = model(x)[1]                                   # [B, 7, 37]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        em, _ = evaluate(model, Xva, yva, gva, device)
        sec = time.time() - t0
        print(f"Epoch {ep}/{args.epochs} — loss {np.mean(losses):.4f} — val EM {em * 100:.2f}% — {sec:.0f}s")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, round(float(np.mean(losses)), 5), round(em, 5), round(sec, 1)])
        if em > best_em:
            best_em, best_epoch = em, ep
            C.save_gplpr(model, args.out, extra={"epoch": ep, "val_exact_match": em})
            print(f"   ↳ lưu checkpoint tốt nhất (epoch {ep})")

    # --- 7. Đo lại bằng checkpoint tốt nhất + tóm tắt
    best_model, _ = C.load_gplpr(args.repo, args.out, device, verbose=False)
    em_after, grp_after = evaluate(best_model, Xva, yva, gva, device)
    gate_ok = em_after >= 0.70
    summary = {
        "exact_match_val_hr_truoc": em_before, "exact_match_val_hr_sau": em_after,
        "theo_nhom_truoc": grp_before, "theo_nhom_sau": grp_after,
        "epoch_tot_nhat": best_epoch, "dong_bang_encoder": bool(args.freeze_encoder),
        "ty_le_tham_so_dong_bang": frozen / total, "so_track_train": len(tr), "so_track_val": len(va),
        "so_anh_cache_train": n, "tong_giay": round(time.time() - t_start, 1),
        "cong_kiem_tra_70_phan_tram": bool(gate_ok), "checkpoint_khoi_dau": info, "tham_so": vars(args),
        "checkpoint_sha256": C.sha256_file(args.out),
    }
    C.write_json(os.path.join(out_dir, "ocr_summary.json"), summary)
    print(f"\nKẾT QUẢ: val EM (ảnh HR) {em_before * 100:.2f}% → {em_after * 100:.2f}%  (epoch tốt nhất: {best_epoch})")
    if not gate_ok:
        print("⚠ CẢNH BÁO: thấp hơn 70% (log cũ đạt ~81%). Hãy kiểm tra tiền xử lý trước khi chạy tiếp!")


if __name__ == "__main__":
    main()
