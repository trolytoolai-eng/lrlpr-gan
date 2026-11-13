# -*- coding: utf-8 -*-
"""
Các lớp Dataset đọc ảnh theo track: cặp LR/HR (1 frame), ảnh HR cho OCR, 5 frame LR, và chồng 5 frame (nhiều frame).
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from lrlpr.datasets.preprocess import (augment_pair, prep_hr_u8, prep_lr_u8, read_rgb, to_chw_float)
from lrlpr.datasets.tracks import frame_path
from lrlpr.utils.constants import N_FRAMES


def worker_init_fn(worker_id):
    """Chạy khi khởi tạo mỗi tiến trình nạp dữ liệu: đặt seed riêng và giới hạn luồng tính toán."""
    torch.set_num_threads(1)
    base = torch.initial_seed() % (2 ** 31)
    random.seed(base)
    np.random.seed(base)


class SRPairDataset(Dataset):
    """
    Dataset cặp (LR, HR) để huấn luyện / kiểm tra mạng làm nét.
    Mỗi mẫu = một track; chọn một frame (ngẫu nhiên hoặc cố định) rồi tiền xử lý.
    Trả về: (ảnh LR 3x16x48, ảnh HR 3x32x96, chuỗi biển số).

    aug=True  -> tăng cường dữ liệu bằng augment_pair(): xoay nhẹ (cả LR+HR), đổi sáng/tương phản,
                 gamma, làm nét nhẹ và sương mù (xác suất fog_p) — các phép đổi ảnh chỉ áp lên LR.
    """

    def __init__(self, rows, frame="random", aug=False, fog_p=0.3):
        self.rows = rows
        self.frame = frame
        self.aug = aug
        self.fog_p = fog_p

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        i = random.randint(1, N_FRAMES) if self.frame == "random" else int(self.frame)
        lr = read_rgb(frame_path(r, "lr", i))
        hr = read_rgb(frame_path(r, "hr", i))
        if self.aug:
            lr, hr = augment_pair(lr, hr, random, self.fog_p)
        lr_t = torch.from_numpy(to_chw_float(prep_lr_u8(lr)))
        hr_t = torch.from_numpy(to_chw_float(prep_hr_u8(hr)))
        return lr_t, hr_t, r["plate_text"]


class HRPlateDataset(Dataset):
    """Dataset ảnh HR (đã CLAHE + đệm + 32x96) và biển số — để huấn luyện / kiểm tra OCR."""

    def __init__(self, rows, frame="random"):
        self.rows = rows
        self.frame = frame

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        i = random.randint(1, N_FRAMES) if self.frame == "random" else int(self.frame)
        hr_u8 = prep_hr_u8(read_rgb(frame_path(r, "hr", i)))
        return torch.from_numpy(hr_u8), r["plate_text"]  # uint8 (32, 96, 3)


class LRFramesDataset(Dataset):
    """Dataset trả về cả 5 frame LR đã tiền xử lý (5x3x16x48) của một track — để chạy mạng làm nét."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        frames = [to_chw_float(prep_lr_u8(read_rgb(frame_path(r, "lr", i))))
                  for i in range(1, N_FRAMES + 1)]
        return torch.from_numpy(np.stack(frames)), idx


class MFPairDataset(Dataset):
    """
    Dataset cho mạng làm nét NHIỀU FRAME: mỗi mẫu = một track.
    Đầu vào: chồng 5 frame LR đã tiền xử lý, ghép theo kênh -> tensor (15, 16, 48).
    Đích   : ảnh HR (CLAHE, 32x96) của frame đứng GIỮA trong thứ tự đưa vào.
    order='random': mỗi lần xoay vòng thứ tự frame ngẫu nhiên (frame nào làm 'tâm' cũng được) để mạng
                    không phụ thuộc vào vị trí cố định; order='fixed': luôn thứ tự 1..5 (dùng cho val).
    Trả về: (chồng LR 15x16x48, ảnh HR 3x32x96, chuỗi biển số).
    """

    def __init__(self, rows, order="random"):
        self.rows = rows
        self.order = order

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        shift = random.randrange(N_FRAMES) if self.order == "random" else 0
        frames = [(shift + k) % N_FRAMES + 1 for k in range(N_FRAMES)]  # số frame (1..5) theo thứ tự đưa vào
        center = frames[N_FRAMES // 2]
        lr = np.concatenate([to_chw_float(prep_lr_u8(read_rgb(frame_path(r, "lr", f)))) for f in frames], axis=0)
        hr = to_chw_float(prep_hr_u8(read_rgb(frame_path(r, "hr", center))))
        return torch.from_numpy(lr), torch.from_numpy(hr), r["plate_text"]
