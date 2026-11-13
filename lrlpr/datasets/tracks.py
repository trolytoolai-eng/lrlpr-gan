# -*- coding: utf-8 -*-
"""
Đọc danh sách track từ file split_*.csv, tìm đường dẫn ảnh, chọn tập con theo nhóm.
"""

import csv
import os
import random
from collections import defaultdict


def read_split_csv(path):
    """
    Đọc file split_*.csv (do notebook 00 tạo). Mỗi dòng là một track với các cột:
    split, scenario, layout, plate_text, track (đường dẫn tương đối trong thư mục dữ liệu).
    """
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _detect_ext(track_dir, kind):
    """Tìm đuôi file (png/jpg) của ảnh '<kind>-001.*' trong một track. Trả về None nếu không có."""
    prefix = f"{kind}-001."
    for name in os.listdir(track_dir):
        if name.startswith(prefix):
            return name[len(prefix):]
    return None


def index_tracks(rows, data_root):
    """
    Bổ sung vào mỗi dòng: đường dẫn thư mục thật ('dir') và đuôi file ảnh LR/HR
    ('ext_lr', 'ext_hr'). Scenario-A dùng .png, Scenario-B dùng .jpg nên phải dò từng track.
    """
    out = []
    for r in rows:
        d = os.path.join(data_root, r["track"])
        r2 = dict(r)
        r2["dir"] = d
        r2["ext_lr"] = _detect_ext(d, "lr")
        r2["ext_hr"] = _detect_ext(d, "hr")
        out.append(r2)
    return out


def frame_path(row, kind, i):
    """Đường dẫn ảnh thứ i (1..5) của loại 'lr' hoặc 'hr' trong track."""
    ext = row["ext_lr"] if kind == "lr" else row["ext_hr"]
    return os.path.join(row["dir"], f"{kind}-{i:03d}.{ext}")


def group_of(row):
    """Tên nhóm của một track, ví dụ 'Scenario-B/Mercosur'."""
    return f'{row["scenario"]}/{row["layout"]}'


def stratified_subset(rows, n, seed=0):
    """
    Lấy ngẫu nhiên (cố định theo seed) n track nhưng GIỮ ĐÚNG TỈ LỆ các nhóm Scenario × layout.
    Dùng để theo dõi val trong lúc huấn luyện. (Lần chạy đầu từng lấy "n track đầu tiên", mà file được
    sắp theo đường dẫn nên toàn bộ là Scenario-A/Brazilian — tức không có ảnh ban đêm nào.)
    """
    if n <= 0 or n >= len(rows):
        return rows
    rng = random.Random(seed)
    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[group_of(r)].append(i)
    picked = []
    for _g, idx in sorted(by_group.items()):
        k = max(1, round(n * len(idx) / len(rows)))
        picked += rng.sample(idx, min(k, len(idx)))
    return [rows[i] for i in sorted(picked)]
