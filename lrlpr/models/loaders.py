# -*- coding: utf-8 -*-
"""
Nạp và lưu mô hình: OCR GP-LPR, mạng làm nét LPSRGAN (1 frame hoặc nhiều frame), mở rộng lớp đầu, đọc biển số.
Cần repo LPSRGAN (đã vá) nằm trong sys.path, xem setup_repo.py.
"""

import os
import sys

import numpy as np
import torch

from lrlpr.utils.constants import DEFAULT_GPLPR_ARGS


def add_repo_to_path(repo_dir):
    """Cho Python thấy các package của repo LPSRGAN (models, datasets, ...)."""
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)


def atomic_torch_save(obj, path):
    """Lưu checkpoint an toàn: ghi ra file tạm rồi đổi tên, tránh file hỏng nếu bị ngắt giữa chừng."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _strip_module_prefix(sd):
    """Bỏ tiền tố 'module.' (do DataParallel thêm vào) nếu có."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}


def load_gplpr(repo_dir, ckpt_path, device, verbose=True):
    """
    Nạp mô hình OCR GP-LPR từ checkpoint. Hỗ trợ 3 dạng file:
      (a) dạng của repo:  {'model': {'name', 'args', 'sd'}, ...}
      (b) dạng {'state_dict': ...}
      (c) state_dict thuần (chỉ gồm tensor)
    Trả về (model, thông_tin). Báo LỖI nếu thiếu / thừa tham số (tránh nạp sai âm thầm).
    """
    add_repo_to_path(repo_dir)
    import models  # package của repo
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = dict(DEFAULT_GPLPR_ARGS)
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict) and "sd" in ck["model"]:
        sd = ck["model"]["sd"]
        args.update(ck["model"].get("args", {}) or {})
        kind = "repo"
    elif isinstance(ck, dict) and "state_dict" in ck:
        sd, kind = ck["state_dict"], "state_dict"
    elif isinstance(ck, dict) and ck and all(torch.is_tensor(v) for v in ck.values()):
        sd, kind = ck, "thuan"
    else:
        keys = list(ck.keys())[:10] if isinstance(ck, dict) else type(ck)
        raise ValueError(f"Không nhận ra định dạng checkpoint {ckpt_path}. Các khóa: {keys}")
    model = models.make({"name": "GPLPR", "args": args})
    res = model.load_state_dict(_strip_module_prefix(sd), strict=False)
    info = {"file": ckpt_path, "dinh_dang": kind, "thieu": len(res.missing_keys),
            "thua": len(res.unexpected_keys), "so_tham_so": sum(p.numel() for p in model.parameters())}
    if verbose:
        print(f"[GP-LPR] nạp {os.path.basename(ckpt_path)} (dạng: {kind}) — thiếu {info['thieu']}, "
              f"thừa {info['thua']} khóa, {info['so_tham_so']:,} tham số")
    if info["thieu"] > 0:
        raise ValueError(f"Checkpoint thiếu {info['thieu']} khóa, ví dụ: {res.missing_keys[:5]}")
    return model.to(device).eval(), info


def save_gplpr(model, path, extra=None):
    """Lưu GP-LPR theo định dạng của repo ({'model': {'name','args','sd'}, ...})."""
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    obj = {"model": {"name": "GPLPR", "args": dict(DEFAULT_GPLPR_ARGS), "sd": sd}}
    if extra:
        obj.update(extra)
    atomic_torch_save(obj, path)


def inflate_first_conv(g, n_frames, center_weight=0.6):
    """
    Biến mạng làm nét 1 frame thành mạng NHIỀU FRAME bằng cách mở rộng lớp tích chập đầu tiên
    từ 3 kênh (1 ảnh RGB) lên 3*n_frames kênh (n ảnh chồng theo kênh).
    Trọng số khởi tạo: frame GIỮA nhận center_weight * W, mỗi frame còn lại nhận
    (1 - center_weight)/(n-1) * W (W = trọng số đã học của mạng 1 frame, tổng các phần = W).
    Nhờ vậy lúc bắt đầu mạng mới hành xử gần giống mạng 1 frame đã tốt rồi mới học tận dụng các frame còn lại.
    """
    old = g.conv_first
    assert old.in_channels == 3, "Mạng đã là nhiều frame"
    new = torch.nn.Conv2d(3 * n_frames, old.out_channels, old.kernel_size, padding=old.padding)
    with torch.no_grad():
        w = old.weight.detach()
        other = (1.0 - center_weight) / (n_frames - 1)
        for k in range(n_frames):
            new.weight[:, 3 * k:3 * k + 3] = w * (center_weight if k == n_frames // 2 else other)
        new.bias.copy_(old.bias.detach())
    g.conv_first = new.to(old.weight.device)
    return g


def load_generator(repo_dir, ckpt_path, device):
    """
    Nạp mạng làm nét LPSRGAN (generator) từ checkpoint theo định dạng repo.
    Dùng load_state_dict(strict=True) để CHẮC CHẮN trọng số đã được nạp đúng
    (hàm models.make của repo nuốt lỗi âm thầm nên không dùng đường đó).
    Nếu checkpoint là mạng NHIỀU FRAME (có khóa 'n_frames' > 1) thì dựng lớp đầu nhiều kênh trước khi nạp.
    """
    add_repo_to_path(repo_dir)
    import models
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    spec = ck["model"]
    args = spec.get("args") or {"num_blocks": 3, "dropout_prob": 0.5}
    g, _d = models.make({"name": "lpsrgan", "args": args})
    n = int(spec.get("n_frames", 1))
    if n > 1:
        g.conv_first = torch.nn.Conv2d(3 * n, g.conv_first.out_channels, g.conv_first.kernel_size,
                                       padding=g.conv_first.padding)
    g.load_state_dict(_strip_module_prefix(spec["sd"]), strict=True)
    return g.to(device).eval(), ck


@torch.no_grad()
def ocr_read(ocr, frames_u8, device, bs=600):
    """
    Cho OCR đọc một loạt ảnh.
    frames_u8: mảng uint8 (N, 32, 96, 3). Trả về (danh sách N chuỗi, mảng N độ tin cậy).
    """
    preds, confs = [], []
    ocr.eval()
    for s in range(0, len(frames_u8), bs):
        x = torch.from_numpy(frames_u8[s:s + bs]).to(device).permute(0, 3, 1, 2).float().div(255.0)
        p, c = ocr.OCR_pred(x)
        preds.extend(p)
        confs.extend(c.tolist())
    return preds, np.array(confs, dtype=np.float32)
