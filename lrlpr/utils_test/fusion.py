# -*- coding: utf-8 -*-
"""
Gộp kết quả của 5 frame (bỏ phiếu) và các độ đo (Exact Match, độ đúng ký tự). Giống utils_test của repo LPSRGAN.
"""

from collections import Counter

import numpy as np


def fuse_mvcp(preds, k):
    """Bỏ phiếu từng ký tự: ở mỗi vị trí chọn ký tự xuất hiện nhiều nhất trong k frame đầu."""
    if k == 1:
        return preds[0]
    return "".join(Counter(chars).most_common(1)[0][0] for chars in zip(*preds[:k]))


def fuse_hc(confs, preds, k):
    """Chọn chuỗi của frame có độ tin cậy cao nhất trong k frame đầu."""
    if k == 1:
        return preds[0]
    c = list(confs[:k])
    return preds[:k][int(np.argmax(c))]


def fuse_mv(confs, preds, k):
    """Chọn chuỗi xuất hiện nhiều lần nhất trong k frame đầu; nếu hòa thì lấy chuỗi tin cậy cao hơn."""
    if k == 1:
        return preds[0]
    s, c = list(preds[:k]), list(confs[:k])
    counts = Counter(s)
    top = max(counts.values())
    best = [x for x, n in counts.items() if n == top]
    if len(best) == 1:
        return best[0]
    idx = max((i for i, x in enumerate(s) if x in best), key=lambda i: c[i])
    return s[idx]


FUSIONS = {"mvcp": lambda p, c, k: fuse_mvcp(p, k),
           "hc": lambda p, c, k: fuse_hc(c, p, k),
           "mv": lambda p, c, k: fuse_mv(c, p, k)}
FUSION_KS = (3, 5)  # gộp 3 frame đầu hoặc cả 5 frame


# ----------------------------------------------------------------------------
# 9. ĐỘ ĐO
# ----------------------------------------------------------------------------
def exact_match(pred, gt):
    """1 nếu đọc đúng TOÀN BỘ biển số, ngược lại 0."""
    return int(pred == gt)


def char_correct(pred, gt):
    """Số ký tự đúng đúng vị trí (so từng vị trí với nhãn thật)."""
    return sum(1 for a, b in zip(pred, gt) if a == b)


def char_accuracy(pred, gt):
    """Tỉ lệ ký tự đúng trên độ dài nhãn thật (0..1)."""
    return char_correct(pred, gt) / max(1, len(gt))
