# -*- coding: utf-8 -*-
"""
Hàm dùng chung khi dạy OCR: Dataset ảnh HR cố định, dựng cache ảnh, đánh giá theo nhóm.
"""

import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lrlpr.utils import api as C


class FixedFrameHRDataset(Dataset):
    """Dataset ảnh HR với frame đã chỉ định sẵn cho từng mẫu — dùng để dựng cache một lần."""

    def __init__(self, items):
        self.items = items  # danh sách (dòng_track, số_frame)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row, f = self.items[i]
        img = C.prep_hr_u8(C.read_rgb(C.frame_path(row, "hr", f)))
        return torch.from_numpy(img), row["plate_text"]


def build_cache(rows, frames_per_track, workers, seed, what):
    """Tiền xử lý ảnh HR của các track và giữ trong bộ nhớ. Trả về (tensor uint8 [N,32,96,3], danh sách nhãn)."""
    rng = random.Random(seed)
    items = []
    for r in rows:
        for f in rng.sample(range(1, C.N_FRAMES + 1), frames_per_track):
            items.append((r, f))
    loader = DataLoader(FixedFrameHRDataset(items), batch_size=256, shuffle=False,
                        num_workers=workers, worker_init_fn=C.worker_init_fn)
    xs, ys, t0 = [], [], time.time()
    for k, (x, y) in enumerate(loader):
        xs.append(x)
        ys.extend(y)
        if k % 20 == 0:
            print(f"   cache {what}: {len(ys):,}/{len(items):,} ảnh — {time.time() - t0:.0f}s")
    X = torch.cat(xs)
    print(f"   cache {what}: xong {len(ys):,} ảnh trong {time.time() - t0:.0f}s")
    return X, ys


def evaluate(model, X, ys, groups, device):
    """Exact Match của OCR trên ảnh HR đã cache. Trả về (EM chung, EM theo nhóm)."""
    preds, _ = C.ocr_read(model, X.numpy(), device)
    ok = np.array([int(p == y) for p, y in zip(preds, ys)])
    by_group = {g: float(ok[[i for i, gg in enumerate(groups) if gg == g]].mean())
                for g in sorted(set(groups))}
    return float(ok.mean()), by_group
