# -*- coding: utf-8 -*-
"""
eval_lib.py — Chạy các nhánh so sánh, đọc biển số bằng OCR, gộp 5 frame và lập bảng kết quả.

Các nhánh (mã nhánh):
  M0  : ảnh LR phóng to thẳng (không làm nét)           — sinh ở Notebook 01
  M1  : bộ làm nét cổ điển (CLAHE + Bilateral + Unsharp) — sinh ở Notebook 01
  HR  : ảnh nét thật (chỉ có ở val)                       — sinh ở Notebook 01, dùng làm cận trên
  M2  : LPSRGAN trọng số công bố (zero-shot)              — sinh ở Notebook 04
  M3  : LPSRGAN fine-tune, dữ liệu nguyên bản             — sinh ở Notebook 04
  M4  : LPSRGAN fine-tune, dữ liệu có tăng cường          — sinh ở Notebook 04

Mỗi nhánh sinh ra:
  <results>/<nhánh>/sr_outputs_<split>.npz  : ảnh sau làm nét của MỌI track (uint8, N x 5 x 32 x 96 x 3)
  <results>/<nhánh>/pred_<split>.csv        : dự đoán từng frame + sau gộp, đúng/sai (mỗi track một dòng)
Nguyên tắc quan trọng: OCR luôn đọc từ chính mảng uint8 đã lưu, nên đọc lại file sẽ cho đúng kết quả cũ.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lrlpr.utils import api as C  # noqa: E402


# Các cách đọc kết quả mỗi track: 1 frame (trung bình 5 frame) hoặc gộp bằng 3 phương pháp bỏ phiếu
FUSION_COLS = ["single"] + [f"{m}_{k}" for m in C.FUSIONS for k in C.FUSION_KS]


# ----------------------------------------------------------------------------
# 1. Nhánh KHÔNG cần mạng: M0 / M1 / HR (chạy trên CPU)
# ----------------------------------------------------------------------------
def build_precomputed_frames(rows, fn, kind="lr", workers=4, label=""):
    """
    Áp hàm fn (ảnh RGB gốc -> ảnh uint8 32x96x3) lên cả 5 frame của mỗi track.
    kind='lr' đọc ảnh LR, kind='hr' đọc ảnh HR. Trả về mảng uint8 (N, 5, 32, 96, 3).
    """
    def one(r):
        return np.stack([fn(C.read_rgb(C.frame_path(r, kind, i))) for i in range(1, C.N_FRAMES + 1)])

    out, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k, arr in enumerate(ex.map(one, rows)):
            out.append(arr)
            if k % 500 == 0:
                print(f"   {label}: {k}/{len(rows)} track — {time.time() - t0:.0f}s")
    print(f"   {label}: xong {len(rows)} track trong {time.time() - t0:.0f}s")
    return np.stack(out)


def save_frames_npz(path, rows, frames):
    """Lưu ảnh của một nhánh (kèm danh sách track) vào file nén .npz."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path, tracks=np.array([r["track"] for r in rows]), frames=frames)


def load_frames_npz(path):
    """Đọc lại file .npz. Trả về (danh sách track, mảng ảnh uint8)."""
    z = np.load(path, allow_pickle=False)
    return [str(t) for t in z["tracks"]], z["frames"]


# ----------------------------------------------------------------------------
# 2. Nhánh dùng mạng làm nét (M2 / M3 / M4)
# ----------------------------------------------------------------------------
@torch.no_grad()
def run_generator(G, rows, device, batch_tracks=64, workers=2):
    """
    Cho mạng làm nét chạy trên 5 frame LR của mỗi track.
    Trả về (mảng uint8 (N,5,32,96,3), số mili-giây GPU cho mỗi ảnh).
    """
    ds = C.LRFramesDataset(rows)
    loader = DataLoader(ds, batch_size=batch_tracks, shuffle=False, num_workers=workers,
                        worker_init_fn=C.worker_init_fn)
    out = np.zeros((len(rows), C.N_FRAMES, C.SR_H, C.SR_W, 3), dtype=np.uint8)
    t_net, t0 = 0.0, time.time()
    G.eval()
    n_in = G.conv_first.in_channels // 3   # 1 = mạng một frame; 5 = mạng nhiều frame
    for k, (x, idx) in enumerate(loader):
        b = x.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        if n_in == 1:
            xin = x.view(-1, 3, C.LR_H, C.LR_W).to(device)     # mỗi frame đi riêng
            t1 = time.time()
            sr = G(xin)
        else:
            # Mạng nhiều frame: chạy 5 lần, mỗi lần một frame làm 'tâm' (xoay vòng thứ tự frame),
            # nên vẫn thu được 5 ảnh cho mỗi track và dùng lại cách gộp bỏ phiếu như cũ.
            x = x.to(device)                                    # (B, 5, 3, 16, 48)
            xin = torch.cat([x[:, [(r + j) % n_in for j in range(n_in)]].reshape(b, 3 * n_in, C.LR_H, C.LR_W)
                             for r in range(C.N_FRAMES)], dim=0)   # (5B, 15, 16, 48), xếp theo r
            t1 = time.time()
            sr = G(xin)                                         # (5B, 3, 32, 96)
            sr = sr.view(C.N_FRAMES, b, 3, C.SR_H, C.SR_W).permute(1, 0, 2, 3, 4).reshape(-1, 3, C.SR_H, C.SR_W)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_net += time.time() - t1
        u8 = (sr.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        out[idx.numpy()] = u8.reshape(b, C.N_FRAMES, C.SR_H, C.SR_W, 3)
        if k % 10 == 0:
            print(f"   làm nét: {min((k + 1) * batch_tracks, len(rows))}/{len(rows)} track — {time.time() - t0:.0f}s")
    return out, t_net / (len(rows) * C.N_FRAMES) * 1000.0


# ----------------------------------------------------------------------------
# 3. Đọc biển số và lập bảng dự đoán
# ----------------------------------------------------------------------------
def read_frames(ocr, frames, device):
    """OCR đọc toàn bộ ảnh. Trả về (chuỗi đọc được (N,5), độ tin cậy (N,5))."""
    n, f = frames.shape[:2]
    preds, confs = C.ocr_read(ocr, frames.reshape(n * f, C.SR_H, C.SR_W, 3), device)
    return np.array(preds, dtype=object).reshape(n, f), confs.reshape(n, f)


def build_pred_df(rows, preds, confs):
    """
    Bảng dự đoán: mỗi track một dòng gồm nhãn thật, chuỗi đọc được ở từng frame (+độ tin cậy),
    kết quả sau khi gộp (3 phương pháp × gộp 3 hoặc 5 frame) và cờ đúng/sai.
    """
    recs = []
    for r, p, c in zip(rows, preds, confs):
        gt = r["plate_text"]
        rec = {"track": r["track"], "scenario": r["scenario"], "layout": r["layout"],
               "group": C.group_of(r), "plate_text": gt}
        for i in range(C.N_FRAMES):
            rec[f"pred_{i + 1}"] = p[i]
            rec[f"conf_{i + 1}"] = round(float(c[i]), 4)
            rec[f"ok_{i + 1}"] = C.exact_match(p[i], gt)
        rec["chars_single"] = float(np.mean([C.char_accuracy(p[i], gt) for i in range(C.N_FRAMES)]))
        for m, fn in C.FUSIONS.items():
            for k in C.FUSION_KS:
                s = fn(list(p), list(c), k)
                rec[f"{m}_{k}"] = s
                rec[f"ok_{m}_{k}"] = C.exact_match(s, gt)
                rec[f"chars_{m}_{k}"] = C.char_accuracy(s, gt)
        recs.append(rec)
    return pd.DataFrame(recs)


def read_pred_csv(path):
    """
    Đọc lại file pred_*.csv. Cột chuỗi (nhãn, biển số đọc được) giữ nguyên là CHỮ — tránh pandas
    tự đổi chuỗi toàn chữ số thành số; chỉ các cột conf_/ok_/chars_ được đổi sang số.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in df.columns:
        if c.startswith(("conf_", "ok_", "chars_")):
            df[c] = pd.to_numeric(df[c])
    return df


def summarize(df, branch, split):
    """Tóm tắt Exact Match và độ đúng ký tự: toàn bộ ('ALL') và từng nhóm Scenario × layout."""
    out = []
    present = list(df["group"].unique())
    # thứ tự: 4 nhóm chuẩn trước, rồi các nhóm khác (ví dụ 'Test/Mercosur' của tập test)
    ordered = [g for g in C.GROUPS if g in present] + sorted(g for g in present if g not in C.GROUPS)
    parts = [("ALL", df)] + [(g, df[df["group"] == g]) for g in ordered]
    for name, sub in parts:
        rec = {"branch": branch, "split": split, "group": name, "n": int(len(sub)),
               "em_single": float(sub[[f"ok_{i}" for i in range(1, C.N_FRAMES + 1)]].to_numpy().mean()),
               "char_single": float(sub["chars_single"].mean())}
        for m in C.FUSIONS:
            for k in C.FUSION_KS:
                rec[f"em_{m}_{k}"] = float(sub[f"ok_{m}_{k}"].mean())
                rec[f"char_{m}_{k}"] = float(sub[f"chars_{m}_{k}"].mean())
        out.append(rec)
    return pd.DataFrame(out)


def evaluate_branch(branch, split, rows, results_dir, ocr, device, repo_dir=None, ckpt=None):
    """
    Đánh giá một nhánh trên một tập (val hoặc test).
      - Có ckpt  : nhánh dùng mạng làm nét -> chạy mạng, lưu ảnh SR vào .npz.
      - Không có : đọc ảnh đã sinh sẵn (.npz) ở Notebook 01.
    Ghi pred_<split>.csv và eval_info_<split>.json. Trả về (bảng tóm tắt, ms/ảnh hoặc None).
    """
    bdir = os.path.join(results_dir, branch)
    os.makedirs(bdir, exist_ok=True)
    npz = os.path.join(bdir, f"sr_outputs_{split}.npz")
    ms, t0 = None, time.time()
    if ckpt:
        G, _ = C.load_generator(repo_dir, ckpt, device)
        frames, ms = run_generator(G, rows, device)
        save_frames_npz(npz, rows, frames)
    else:
        tracks, frames = load_frames_npz(npz)
        assert tracks == [r["track"] for r in rows], f"{branch}/{split}: danh sách track không khớp split!"
    preds, confs = read_frames(ocr, frames, device)
    df = build_pred_df(rows, preds, confs)
    df.to_csv(os.path.join(bdir, f"pred_{split}.csv"), index=False)
    info = {"nhanh": branch, "tap": split, "so_track": len(rows), "ckpt": ckpt,
            "ckpt_sha256": C.sha256_file(ckpt) if ckpt else None, "ms_moi_anh_mang_lam_net": ms,
            "npz_sha256": C.sha256_file(npz), "ket_thuc": C.now_iso(), "giay": round(time.time() - t0, 1)}
    C.write_json(os.path.join(bdir, f"eval_info_{split}.json"), info)
    summ = summarize(df, branch, split)
    row = summ[summ["group"] == "ALL"].iloc[0]
    print(f"   [{branch} · {split}] Exact Match: 1 frame {row['em_single'] * 100:.1f}% | "
          f"gộp 5 frame (bỏ phiếu ký tự) {row['em_mvcp_5'] * 100:.1f}%")
    return summ, ms


def evaluate_with_ocr(branch, src_branch, split, rows, results_dir, ocr, device, ocr_ckpt=None):
    """
    Đọc lại ảnh ĐÃ CÓ của nhánh src_branch bằng MỘT OCR KHÁC (ví dụ OCR đã dạy lại trên ảnh GAN),
    rồi ghi kết quả thành nhánh mới tên 'branch' (ví dụ 'M4+' = ảnh của M4, đọc bằng OCR mới).
    Không chạy lại mạng làm nét, chỉ đọc lại ảnh trong sr_outputs_<split>.npz của src_branch.
    """
    bdir = os.path.join(results_dir, branch)
    os.makedirs(bdir, exist_ok=True)
    npz = os.path.join(results_dir, src_branch, f"sr_outputs_{split}.npz")
    tracks, frames = load_frames_npz(npz)
    assert tracks == [r["track"] for r in rows], f"{src_branch}/{split}: danh sách track không khớp split!"
    t0 = time.time()
    preds, confs = read_frames(ocr, frames, device)
    df = build_pred_df(rows, preds, confs)
    df.to_csv(os.path.join(bdir, f"pred_{split}.csv"), index=False)
    C.write_json(os.path.join(bdir, f"eval_info_{split}.json"), {
        "nhanh": branch, "tap": split, "so_track": len(rows), "anh_lay_tu_nhanh": src_branch,
        "ocr_ckpt": ocr_ckpt, "ocr_sha256": C.sha256_file(ocr_ckpt) if ocr_ckpt else None,
        "npz_nguon_sha256": C.sha256_file(npz), "ket_thuc": C.now_iso(), "giay": round(time.time() - t0, 1)})
    summ = summarize(df, branch, split)
    row = summ[summ["group"] == "ALL"].iloc[0]
    print(f"   [{branch} · {split}] Exact Match: 1 frame {row['em_single'] * 100:.1f}% | "
          f"gộp 5 frame (bỏ phiếu ký tự) {row['em_mvcp_5'] * 100:.1f}%")
    return summ


# ----------------------------------------------------------------------------
# 4. Chọn mô hình / cách gộp trên VAL và khóa test
# ----------------------------------------------------------------------------
def choose_on_val(summary, candidates):
    """
    Chọn CÁCH GỘP và MÔ HÌNH tốt nhất chỉ dựa trên tập val (nhóm 'ALL'):
      - cách gộp  : cột có Exact Match trung bình cao nhất trên các nhánh ứng viên
      - mô hình   : nhánh có Exact Match cao nhất theo cách gộp đó
    summary: bảng gộp từ summarize(); candidates: danh sách mã nhánh được phép chọn (không gồm HR).
    """
    v = summary[(summary["split"] == "val") & (summary["group"] == "ALL") &
                (summary["branch"].isin(candidates))]
    cols = {c: f"em_{c}" for c in FUSION_COLS}
    means = {c: float(v[col].mean()) for c, col in cols.items()}
    fusion = max(means, key=means.get)
    col = cols[fusion]
    best = v.loc[v[col].idxmax()]
    return {"cach_gop_tot_nhat": fusion, "mo_hinh_tot_nhat": str(best["branch"]),
            "exact_match_val_cua_mo_hinh_tot_nhat": float(best[col]),
            "trung_binh_theo_cach_gop": means,
            "bang_val": v[["branch"] + list(cols.values())].round(4).to_dict(orient="records"),
            "thoi_diem_chot": C.now_iso()}


def acquire_test_lock(evidence_dir, chosen_path, lock_name="test_run.lock"):
    """
    Khóa test: chỉ cho chạy test MỘT lần (mỗi vòng thí nghiệm một khóa riêng) và chỉ sau khi đã chốt
    lựa chọn trên val. Ghi evidence/<lock_name> (giờ chạy + nội dung lựa chọn) làm bằng chứng không "nhìn trước" test.
    Vòng 1 dùng 'test_run.lock'; vòng 2 dùng 'test_run_v2.lock'.
    """
    lock = os.path.join(evidence_dir, lock_name)
    if os.path.exists(lock):
        raise RuntimeError(f"Test đã được chạy trước đó (xem {lock}). Không chạy lần hai để giữ kết quả khách quan.")
    if not os.path.exists(chosen_path):
        raise RuntimeError("Chưa có chosen_on_val.json — phải chốt mô hình trên val TRƯỚC khi chạy test.")
    C.write_json(lock, {"bat_dau_test": C.now_iso(), "chosen_on_val": C.read_json(chosen_path),
                        "chosen_on_val_sha256": C.sha256_file(chosen_path)})
    print(f"🔒 Đã khóa test: {lock}")
