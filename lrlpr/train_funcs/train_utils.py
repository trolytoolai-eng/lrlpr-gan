# -*- coding: utf-8 -*-
"""
Hàm phụ cho vòng huấn luyện mạng làm nét: kiểm tra trên val, GradScaler, learning rate cosine.
"""

import math

import numpy as np
import torch

from lrlpr.utils import api as C


@torch.no_grad()
def validate(G, loader, ocr, device):
    """Tính L1, PSNR và (nếu có OCR) Exact Match 1 frame của ảnh SR trên tập val."""
    G.eval()
    l1s, psnrs, ok, n = [], [], 0, 0
    for lr, hr, gt in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = G(lr)
        l1s.append(torch.mean(torch.abs(sr - hr)).item())
        srq = sr.clamp(0, 1)
        mse = torch.mean((srq - hr) ** 2, dim=(1, 2, 3)).clamp_min(1e-10)
        psnrs.extend((10 * torch.log10(1.0 / mse)).tolist())
        if ocr is not None:
            u8 = (srq * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            preds, _ = C.ocr_read(ocr, u8, device)
            ok += sum(int(p == g) for p, g in zip(preds, gt))
        n += len(gt)
    return {"val_l1": float(np.mean(l1s)), "val_psnr": float(np.mean(psnrs)),
            "val_em": (ok / n) if ocr is not None else float("nan")}


def make_scaler(enabled):
    """GradScaler tương thích nhiều phiên bản PyTorch."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def cosine_lr(step, total, lr0, ratio):
    """Learning rate giảm dần theo hình cosine từ lr0 xuống lr0*ratio."""
    t = min(1.0, step / max(1, total))
    return lr0 * (ratio + (1 - ratio) * 0.5 * (1 + math.cos(math.pi * t)))
