# -*- coding: utf-8 -*-
"""
train_ft.py — Fine-tune mạng làm nét LPSRGAN (Notebook 03). Thay cho train.py của repo.

VÌ SAO KHÔNG DÙNG train.py CỦA REPO?
  train.py gốc có nhiều chỗ làm hỏng / làm chậm trên Colab: ép CUDA_VISIBLE_DEVICES="1",
  bật set_detect_anomaly(True) (rất chậm), num_workers=0, chạy vòng pre-train 1000 lượt ở epoch 1,
  vẽ ảnh mỗi 3 bước, gửi email cuối cùng, và cần OCR Keras không có sẵn.

LOSS (giống hành vi THỰC của repo, cộng thêm phần tuỳ chọn):
  Loss của generator = L1 + VGG19 perceptual.
  (Trong code gốc, phần adversarial và OCR loss không tạo ra gradient nên không ảnh hưởng generator.)
  --ocr_weight > 0 : thêm LOSS OCR CÓ GRADIENT THẬT (cross-entropy của OCR đóng băng) như bài báo
                     LPSRGAN / LCOFL — ép mạng làm nét vẽ ra chữ mà OCR đọc được. Mặc định 0 (tắt).
  --adv_weight > 0 : thêm thành phần adversarial (LSGAN) như bài báo — mặc định 0 (tắt).

NHIỀU FRAME (--mf 1):
  Thay vì làm nét từng frame riêng, đưa chồng 5 frame LR của một track vào mạng cùng lúc (15 kênh).
  Mỗi frame mang thêm chi tiết khác nhau nên mạng có thể khôi phục chữ mà một frame đơn lẻ không đủ thông tin.
  Lớp đầu của mạng 1 frame đã học được "mở rộng" (inflate_first_conv) để khởi đầu từ mạng tốt sẵn có.

TĂNG CƯỜNG DỮ LIỆU (--aug 1):
  xoay nhẹ ±2° (cả LR+HR, cùng góc), đổi sáng/tương phản, gamma, làm nét nhẹ và sương mù — các phép
  đổi ảnh chỉ áp lên LR (xem lrlpr_common.augment_pair; không dùng augment của repo vì repo làm hỏng ảnh HR).

Đầu ra trong thư mục --out:
  best.pth, last.pth        : checkpoint (định dạng repo, nạp được bằng lrlpr_common.load_generator)
  train_log.csv             : log từng epoch (loss, PSNR, Exact Match 1 frame, thời gian)
  run_config.json           : toàn bộ tham số + mã băm (bằng chứng)
  calibration.json          : (chỉ khi dùng --calibrate) tốc độ đo được
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lrlpr.losses.perceptual import VGG19Perceptual
from lrlpr.train_funcs.train_utils import cosine_lr, make_scaler, validate
from lrlpr.utils import api as C
from lrlpr.utils.config import apply_yaml_defaults


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--init", required=True, help="trọng số LPSRGAN để bắt đầu (best_model_Epoch_116.pth)")
    ap.add_argument("--out", required=True, help="thư mục lưu kết quả của lần chạy này")
    ap.add_argument("--ocr_ckpt", default="", help="OCR đã fine-tune, dùng để đo Exact Match trên val")
    ap.add_argument("--aug", type=int, default=0)
    ap.add_argument("--fog_p", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps_per_epoch", type=int, default=0, help="0 = đi hết một lượt dữ liệu")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr_min_ratio", type=float, default=0.1)
    ap.add_argument("--adv_weight", type=float, default=0.0)
    ap.add_argument("--ocr_weight", type=float, default=0.0,
                    help="trọng số loss OCR có gradient (0 = tắt). Cần --ocr_ckpt")
    ap.add_argument("--mf", type=int, default=0,
                    help="1 = mạng NHIỀU FRAME: đưa chồng 5 frame LR vào cùng lúc (xem MFPairDataset)")
    ap.add_argument("--mf_center_weight", type=float, default=0.6,
                    help="trọng số khởi tạo của frame giữa khi mở rộng lớp đầu (các frame còn lại chia phần còn lại)")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_tracks", type=int, default=500)
    ap.add_argument("--max_train_tracks", type=int, default=0, help="0 = dùng hết (chỉ để chạy thử)")
    ap.add_argument("--max_minutes", type=float, default=0, help="0 = không giới hạn")
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--vgg_pretrained", type=int, default=1)
    ap.add_argument("--calibrate", type=int, default=0, help=">0: chỉ đo tốc độ N bước rồi thoát")
    ap.add_argument("--seed", type=int, default=42)
    apply_yaml_defaults(ap)
    args = ap.parse_args()

    C.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp) and device.type == "cuda"
    os.makedirs(args.out, exist_ok=True)
    print(f"Thiết bị: {device} | AMP: {use_amp} | tăng cường: {bool(args.aug)} | seed {args.seed}")
    C.add_repo_to_path(args.repo)

    # --- 1. Dữ liệu
    tr = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_train.csv")), args.data_root)
    va = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, "split_val.csv")), args.data_root)
    if args.max_train_tracks:
        tr = tr[:args.max_train_tracks]
    # Val theo dõi lúc huấn luyện: lấy NGẪU NHIÊN nhưng đủ tỉ lệ 4 nhóm (xem C.stratified_subset)
    va = C.stratified_subset(va, args.val_tracks, args.seed) if args.val_tracks else va
    if args.mf:
        # Mạng NHIỀU FRAME: đầu vào là chồng 5 frame LR (15 kênh), đích là HR của frame giữa
        if args.aug:
            raise SystemExit("--mf chưa hỗ trợ --aug (tăng cường) — hãy dùng --aug 0.")
        train_ds = C.MFPairDataset(tr, order="random")
        val_ds = C.MFPairDataset(va, order="fixed")
    else:
        train_ds = C.SRPairDataset(tr, frame="random", aug=bool(args.aug), fog_p=args.fog_p)
        val_ds = C.SRPairDataset(va, frame=1, aug=False)
    mk = dict(num_workers=args.workers, worker_init_fn=C.worker_init_fn,
              pin_memory=(device.type == "cuda"), persistent_workers=(args.workers > 0))
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, **mk)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, **mk)
    spe = args.steps_per_epoch or len(train_loader)
    spe = min(spe, len(train_loader))
    print(f"Train {len(tr):,} track | val {len(va):,} track | {spe} bước/epoch")

    # --- 2. Mô hình: nạp trọng số công bố, kiểm tra đã nạp ĐÚNG (strict) rồi mới huấn luyện
    G, ck = C.load_generator(args.repo, args.init, device)
    g_args = ck["model"].get("args") or {"num_blocks": 3, "dropout_prob": 0.5}
    w0 = next(iter(G.state_dict().values())).detach().cpu().clone()
    assert torch.equal(w0, next(iter(ck["model"]["sd"].values())).cpu()), "Nạp trọng số khởi đầu KHÔNG khớp!"
    if args.mf:
        # Mở rộng lớp đầu từ 1 frame (3 kênh) lên 5 frame (15 kênh); frame giữa giữ phần lớn trọng số cũ
        C.inflate_first_conv(G, C.N_FRAMES, args.mf_center_weight)
        G = G.to(device)
        print(f"Đã mở rộng thành mạng {C.N_FRAMES} frame (trọng số frame giữa {args.mf_center_weight})")
    print(f"Đã nạp trọng số khởi đầu: {os.path.basename(args.init)} ({sum(p.numel() for p in G.parameters()):,} tham số)")
    ocr = None
    if args.ocr_weight > 0 and not args.ocr_ckpt:
        raise SystemExit("--ocr_weight > 0 cần có --ocr_ckpt (OCR dùng để tính loss).")
    if args.ocr_ckpt:
        ocr, _ = C.load_gplpr(args.repo, args.ocr_ckpt, device)
        for p in ocr.parameters():
            p.requires_grad = False  # OCR đóng băng: chỉ để chấm điểm, gradient vẫn chảy QUA nó về mạng làm nét
    vgg = VGG19Perceptual(pretrained=bool(args.vgg_pretrained)).to(device).eval()
    l1_fn = nn.L1Loss()
    opt = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.555))  # betas giống config repo
    scaler = make_scaler(use_amp)
    D = opt_d = None
    if args.adv_weight > 0:
        import models
        _, D = models.make({"name": "lpsrgan", "args": g_args})
        D = D.to(device)
        opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.555))

    # --- 3. Chế độ đo tốc độ (hiệu chỉnh)
    def step_fn(lr, hr, gt, lr_now):
        """Một bước huấn luyện. Trả về (loss tổng, loss L1, loss VGG, loss OCR) hoặc None nếu loss không hữu hạn."""
        for g in opt.param_groups:
            g["lr"] = lr_now
        lr, hr = lr.to(device, non_blocking=True), hr.to(device, non_blocking=True)
        G.train()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            sr = G(lr)
            l1 = l1_fn(sr, hr)
            pv = vgg(sr, hr)
            loss = l1 + pv
            if D is not None:  # thành phần adversarial (LSGAN), chỉ khi bật
                loss = loss + args.adv_weight * torch.mean((torch.mean(D(sr)) - 1) ** 2)
        ocr_l = None
        if args.ocr_weight > 0:
            # LOSS OCR CÓ GRADIENT THẬT (khác với repo gốc, nơi OCR loss chỉ là con số đếm sai):
            # cho ảnh SR đi qua OCR đã đóng băng rồi tính cross-entropy với biển số thật. Gradient chảy ngược
            # qua OCR về mạng làm nét, ép nó vẽ chữ mà OCR ĐỌC ĐƯỢC. Chạy ở fp32 cho an toàn.
            with torch.autocast(device_type=device.type, enabled=False):
                srf = sr.float()
                x = srf + (srf.clamp(0, 1) - srf).detach()   # giá trị bị kẹp vào [0,1], gradient vẫn đi thẳng qua
                logits = ocr(x)[1]                           # [B, 7, 37]
                tgt = ocr.converter.encode_list(list(gt), K=C.PLATE_LEN).to(device)
                ocr_l = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
            loss = loss + args.ocr_weight * ocr_l
        if not torch.isfinite(loss):
            return None
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if D is not None:
            d_loss = torch.mean((D(hr) - 1) ** 2) + torch.mean(D(sr.detach().float()) ** 2)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        return (loss.item(), l1.item(), float(pv.detach()),
                float(ocr_l.detach()) if ocr_l is not None else 0.0)

    if args.calibrate > 0:
        print(f"[Đo tốc độ] chạy {args.calibrate} bước ...")
        times, it = [], iter(train_loader)
        for k in range(args.calibrate):
            t0 = time.time()
            try:
                lr, hr, gt = next(it)
            except StopIteration:  # hết một lượt dữ liệu thì bắt đầu lượt mới
                it = iter(train_loader)
                lr, hr, gt = next(it)
            t_data = time.time() - t0
            r = step_fn(lr, hr, gt, args.lr)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - t0, t_data))
            if r is None:
                print("  bước", k, "loss không hữu hạn!")
        used = times[5:] if len(times) > 8 else times  # bỏ vài bước đầu (khởi động)
        sec_it = float(np.mean([t for t, _ in used]))
        res = {"giay_moi_buoc": sec_it, "giay_cho_du_lieu_moi_buoc": float(np.mean([d for _, d in used])),
               "buoc_moi_epoch": spe, "giay_moi_epoch_uoc_tinh": sec_it * spe, "so_buoc_do": len(times),
               "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
        C.write_json(os.path.join(args.out, "calibration.json"), res)
        print(f"  {sec_it:.3f} giây/bước (chờ dữ liệu {res['giay_cho_du_lieu_moi_buoc']:.3f}s) "
              f"→ ước tính {res['giay_moi_epoch_uoc_tinh']:.0f} giây/epoch ({spe} bước)")
        return

    # --- 4. Chuẩn bị ghi log + tiếp tục từ checkpoint (nếu cần)
    log_path = os.path.join(args.out, "train_log.csv")
    best_path, last_path = os.path.join(args.out, "best.pth"), os.path.join(args.out, "last.pth")
    start_epoch, best_metric, step = 1, -1e9, 0
    if args.resume and os.path.exists(last_path):
        rc = torch.load(last_path, map_location="cpu", weights_only=False)
        G.load_state_dict(rc["model"]["sd"])
        opt.load_state_dict(rc["optimizer"])
        start_epoch, best_metric, step = rc["epoch"] + 1, rc["best_metric"], rc["step"]
        print(f"Tiếp tục từ epoch {start_epoch} (đã chạy {step} bước)")
    else:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "buoc", "train_loss", "train_l1", "train_vgg", "train_ocr", "val_l1",
                                    "val_psnr", "val_exact_match_1frame", "lr_cuoi", "giay"])
        C.write_json(os.path.join(args.out, "run_config.json"), {
            "tham_so": vars(args), "bat_dau": C.now_iso(), "torch": torch.__version__,
            "init_sha256": C.sha256_file(args.init), "g_args": g_args,
            "ghi_chu": "loss = L1 + VGG19 perceptual (+ ocr_weight * CE của OCR đóng băng, + adversarial nếu bật)"})

    # --- 5. Số liệu mốc trước khi huấn luyện (để thấy fine-tune có cải thiện không)
    if start_epoch == 1:
        m0 = validate(G, val_loader, ocr, device)
        print(f"[Trước fine-tune] val L1 {m0['val_l1']:.4f} | PSNR {m0['val_psnr']:.2f} dB | "
              f"EM 1 frame {m0['val_em'] * 100:.2f}%")
        C.write_json(os.path.join(args.out, "val_truoc_finetune.json"), m0)
        # Lưu ý: best.pth là epoch tốt nhất TRONG LÚC fine-tune (không so với mốc zero-shot ở trên)

    # --- 6. Vòng huấn luyện
    total_steps = args.epochs * spe
    t_run = time.time()
    bad = 0
    for ep in range(start_epoch, args.epochs + 1):
        t_ep, sums, cnt = time.time(), np.zeros(4), 0
        for k, (lr, hr, gt) in enumerate(train_loader):
            if k >= spe:
                break
            lr_now = cosine_lr(step, total_steps, args.lr, args.lr_min_ratio)
            r = step_fn(lr, hr, gt, lr_now)
            if r is None:  # loss bị NaN/inf -> bỏ qua bước này
                bad += 1
                if bad > 50:
                    raise RuntimeError("Loss không hữu hạn quá 50 lần — dừng để kiểm tra.")
                continue
            sums += r
            cnt += 1
            step += 1
            if k % 100 == 0:
                done = (ep - 1) * spe + k + 1
                el = time.time() - t_run
                eta = el / max(1, done - (start_epoch - 1) * spe) * (total_steps - done)
                print(f"  epoch {ep}/{args.epochs} bước {k}/{spe} — loss {r[0]:.4f} — "
                      f"đã {C.fmt_seconds(el)}, còn khoảng {C.fmt_seconds(eta)}")
        m = validate(G, val_loader, ocr, device)
        avg = sums / max(1, cnt)
        sec = time.time() - t_ep
        metric = m["val_em"] if ocr is not None else -m["val_l1"]
        print(f"Epoch {ep}/{args.epochs}: train loss {avg[0]:.4f} (OCR {avg[3]:.3f}) | val L1 {m['val_l1']:.4f} | "
              f"PSNR {m['val_psnr']:.2f} | EM 1 frame {m['val_em'] * 100:.2f}% | {sec:.0f}s")
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([ep, step, round(avg[0], 5), round(avg[1], 5), round(avg[2], 5), round(avg[3], 5),
                                    round(m["val_l1"], 5), round(m["val_psnr"], 3),
                                    round(m["val_em"], 5), opt.param_groups[0]["lr"], round(sec, 1)])
        spec = {"name": "lpsrgan", "args": g_args, "n_frames": C.N_FRAMES if args.mf else 1,
                "sd": {k2: v.detach().cpu() for k2, v in G.state_dict().items()}}
        # last.pth giữ thêm optimizer để có thể tiếp tục khi bị ngắt; best.pth chỉ cần mô hình
        best_now = max(best_metric, metric)
        C.atomic_torch_save({"model": spec, "epoch": ep, "metrics": m, "optimizer": opt.state_dict(),
                             "best_metric": best_now, "step": step, "train_args": vars(args)}, last_path)
        if metric >= best_metric:  # '>=': nếu hòa thì lấy epoch sau (đã huấn luyện nhiều hơn)
            best_metric = metric
            C.atomic_torch_save({"model": spec, "epoch": ep, "metrics": m, "train_args": vars(args)}, best_path)
            print("   ↳ lưu best.pth")
        if args.max_minutes and (time.time() - t_run) / 60 > args.max_minutes:
            print("Đã quá --max_minutes, dừng sớm.")
            break

    if not os.path.exists(best_path):  # chưa epoch nào cải thiện so với mốc đầu -> dùng bản cuối
        C.atomic_torch_save(torch.load(last_path, map_location="cpu", weights_only=False), best_path)
    res = C.read_json(os.path.join(args.out, "run_config.json"))
    res.update({"ket_thuc": C.now_iso(), "tong_giay": round(time.time() - t_run, 1),
                "best_sha256": C.sha256_file(best_path), "so_buoc": step})
    C.write_json(os.path.join(args.out, "run_config.json"), res)
    print(f"\nXONG. Tổng {C.fmt_seconds(time.time() - t_run)}. Checkpoint: {best_path}")


if __name__ == "__main__":
    main()
