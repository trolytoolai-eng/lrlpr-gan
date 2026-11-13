# -*- coding: utf-8 -*-
"""
Tiền xử lý ảnh (đệm tỉ lệ 3:1, đổi kích thước, CLAHE cho ảnh HR), tăng cường dữ liệu và bộ làm nét cổ điển (M1).
Giống hệt cách làm của repo LPSRGAN.
"""

import math

import cv2
import numpy as np
import torch
from PIL import Image

from lrlpr.utils.constants import (ASPECT, ASPECT_TOL, LR_H, LR_W, PAD_COLOR, SR_H, SR_W)


def read_rgb(path):
    """Đọc ảnh và đổi sang RGB (OpenCV mặc định đọc BGR)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pad_to_ratio(img, ratio=ASPECT, tol=ASPECT_TOL, color=PAD_COLOR):
    """
    Đệm viền xám để tỉ lệ rộng/cao nằm trong [ratio - tol, ratio + tol], KHÔNG làm méo ký tự.
    Kết quả giống hàm pad_with_mask của repo (nhưng tính trực tiếp, không dùng vòng lặp từng pixel).
    """
    h, w = img.shape[:2]
    lo, hi = ratio - tol, ratio + tol
    if lo <= w / h <= hi:
        return img
    bw = bh = 0
    if w / h < lo:  # ảnh quá "vuông": đệm thêm hai bên trái/phải
        bw = max(0, int(math.ceil(lo * h - w)))
        while (w + bw) / h < lo:
            bw += 1
        while bw > 0 and (w + bw - 1) / h >= lo:
            bw -= 1
    else:           # ảnh quá "dài": đệm thêm hai bên trên/dưới
        bh = max(0, int(math.ceil(w / hi - h)))
        while w / (h + bh) > hi:
            bh += 1
        while bh > 0 and w / (h + bh - 1) <= hi:
            bh -= 1
    bw //= 2   # chia đều hai phía (đúng như repo)
    bh //= 2
    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=color)


def _bicubic():
    """Hằng số nội suy bicubic của Pillow (tương thích nhiều phiên bản)."""
    return getattr(getattr(Image, "Resampling", Image), "BICUBIC")


def resize_u8(img, out_h, out_w):
    """Đổi kích thước bằng nội suy bicubic (Pillow), giữ kiểu uint8 — giống torchvision.Resize của repo."""
    return np.array(Image.fromarray(img).resize((out_w, out_h), _bicubic()))  # np.array = bản sao ghi được


def to_chw_float(img_u8):
    """Ảnh uint8 (cao, rộng, 3) -> mảng float32 (3, cao, rộng) trong khoảng [0, 1]."""
    return (img_u8.astype(np.float32) / 255.0).transpose(2, 0, 1).copy()


def clahe_hr(img_u8, clip=4.0, grid=(2, 2)):
    """
    Cân bằng histogram thích ứng (CLAHE) bằng kornia — đúng như repo áp lên ảnh HR mục tiêu.
    Mục đích: ảnh HR có độ tương phản đồng đều hơn để mạng làm nét học theo.
    """
    import kornia
    t = torch.from_numpy(np.ascontiguousarray(img_u8)).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
    out = kornia.enhance.equalize_clahe(t, clip_limit=clip, grid_size=grid)
    return out.mul(255.0).byte()[0].permute(1, 2, 0).contiguous().numpy()


def prep_lr_u8(img):
    """Ảnh LR gốc -> đệm tỉ lệ -> thu về 16x48 (đầu vào của mạng làm nét). Trả về uint8."""
    return resize_u8(pad_to_ratio(img), LR_H, LR_W)


def prep_hr_u8(img):
    """Ảnh HR gốc -> CLAHE -> đệm tỉ lệ -> 32x96 (ảnh đích của mạng làm nét). Trả về uint8."""
    return resize_u8(pad_to_ratio(clahe_hr(img)), SR_H, SR_W)


def prep_plain_u8(img):
    """Ảnh gốc -> đệm tỉ lệ -> 32x96 (KHÔNG làm nét). Dùng cho mốc M0."""
    return resize_u8(pad_to_ratio(img), SR_H, SR_W)


def _rotate(img, angle):
    """Xoay ảnh quanh tâm một góc nhỏ (độ), phần viền thiếu được lấp bằng pixel biên."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def augment_pair(lr, hr, rng, fog_p=0.3):
    """
    TĂNG CƯỜNG DỮ LIỆU cho một cặp (LR, HR). Mỗi lần chọn NGẪU NHIÊN một phép (giống tinh thần augment của repo):
      0  xoay nhẹ ±2°  — áp cho CẢ LR và HR với CÙNG một góc (để hai ảnh vẫn khớp nhau)
      1  đổi độ sáng / tương phản   — chỉ áp lên LR
      2  đổi gamma (0.8 – 1.2)       — chỉ áp lên LR
      3  làm nét nhẹ (unsharp mask)  — chỉ áp lên LR
      4  giữ nguyên (xác suất nhỏ)
    Sau đó, với xác suất fog_p, thêm sương mù lên LR (theo gợi ý của thầy).
    Chỉ biến đổi LR vì ảnh HR là đáp án thật: nó không nên bị làm xấu hay đổi màu.

    GHI CHÚ: không dùng hàm augment của repo vì nó có lỗi — khi chọn phép xoay (SafeRotate) ảnh HR bị
    co về kích thước ảnh LR (ví dụ 45x130 thành 20x57), làm cặp huấn luyện bị hỏng và HR bị mờ.
    rng: module random hoặc random.Random.
    """
    choice = rng.choices(range(5), weights=[1, 1, 1, 1, 0.1])[0]
    if choice == 0:
        ang = rng.uniform(-2.0, 2.0)
        lr, hr = _rotate(lr, ang), _rotate(hr, ang)
    elif choice == 1:
        alpha = 1.0 + rng.uniform(-0.2, 0.2)     # tương phản
        beta = 1.0 + rng.uniform(-0.1, 0.1)      # độ sáng
        lr = np.clip(lr.astype(np.float32) * alpha * beta, 0, 255).astype(np.uint8)
    elif choice == 2:
        gamma = rng.uniform(0.8, 1.2)
        lut = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255).astype(np.uint8)
        lr = cv2.LUT(lr, lut)
    elif choice == 3:
        a = rng.uniform(0.2, 0.5)
        blur = cv2.GaussianBlur(lr, (3, 3), 0)
        lr = np.clip(lr.astype(np.float32) + a * (lr.astype(np.float32) - blur), 0, 255).astype(np.uint8)
    if rng.random() < fog_p:
        lr = add_fog(lr, rng)
    return lr, hr


def add_fog(img, rng):
    """
    Thêm 'sương mù' nhẹ vào ảnh LR (tăng cường dữ liệu theo gợi ý của thầy):
    pha ảnh với một lớp sáng đều rồi làm mờ rất nhẹ -> giảm tương phản như khi trời sương.
    rng là đối tượng random.Random để kết quả lặp lại được.
    """
    alpha = rng.uniform(0.10, 0.35)        # độ dày sương
    airlight = rng.uniform(200, 255)       # độ sáng của lớp sương
    fogged = img.astype(np.float32) * (1 - alpha) + airlight * alpha
    fogged = cv2.GaussianBlur(fogged, (3, 3), 0.5)
    return np.clip(fogged, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# 5. BỘ LÀM NÉT CỔ ĐIỂN (CV) — mốc M1, tham số lấy từ run_cv_baseline_experiment.py
# ----------------------------------------------------------------------------
def cv_pipeline_rgb(img_rgb):
    """
    Pipeline thị giác máy tính cổ điển: CLAHE -> lọc Bilateral -> Unsharp Mask.
    Đầu vào / đầu ra: ảnh RGB uint8 (bên trong đổi sang BGR cho giống code thực nghiệm cũ).
    """
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Bước 1: CLAHE trên kênh sáng (L) để kéo sáng vùng tối
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(l)
    bgr = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # Bước 2: khử nhiễu hạt nhưng giữ viền chữ
    bgr = cv2.bilateralFilter(bgr, d=5, sigmaColor=30, sigmaSpace=30)
    # Bước 3: làm nổi viền chữ bằng Unsharp Mask
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0.8)
    sharp = 2.8 * bgr.astype(np.float32) - 1.8 * blurred.astype(np.float32)
    bgr = np.clip(sharp, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
