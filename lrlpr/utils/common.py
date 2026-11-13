# -*- coding: utf-8 -*-
"""
Tiện ích chung: seed, giờ, đọc ghi JSON, mã băm, chạy lệnh con, giải nén dữ liệu, đo thời gian từng bước,
lưu thông tin môi trường (bằng chứng).
"""

import csv
import datetime
import hashlib
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import torch


def set_seed(seed):
    """Đặt seed cho random / numpy / torch để kết quả lặp lại được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def now_iso():
    """Giờ hiện tại dạng chuỗi, ví dụ '2026-11-13T09:30:15'."""
    return datetime.datetime.now().isoformat(timespec="seconds")


def fmt_seconds(s):
    """Đổi số giây thành dạng 'mm:ss' cho dễ đọc."""
    s = int(round(s))
    return f"{s // 60:02d}:{s % 60:02d}"


def write_json(path, obj):
    """Ghi dict ra file JSON (tự tạo thư mục nếu chưa có)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    """Đọc file JSON thành dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    """Mã băm sha256 của một file — dùng làm bằng chứng file không bị sửa."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_script(cmd, env_extra=None):
    """
    Chạy một lệnh con (ví dụ: python train_ocr.py ...) và IN OUTPUT RA NGAY khi nó xuất hiện,
    để thấy tiến độ trong notebook. Nếu lệnh lỗi sẽ báo kèm 30 dòng cuối để dễ tìm nguyên nhân.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", NO_ALBUMENTATIONS_UPDATE="1")
    env.update(env_extra or {})
    p = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env, bufsize=1)
    tail = []
    for line in p.stdout:
        print(line, end="")
        tail.append(line)
        tail = tail[-30:]
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"Lệnh thất bại (mã thoát {rc}). 30 dòng cuối:\n" + "".join(tail))


def find_data_root(base):
    """Tìm thư mục chứa cả train/ và test/ (sau khi giải nén có thể nằm sâu thêm 1–2 cấp)."""
    base_depth = base.rstrip("/\\").count(os.sep)
    for r, dirs, _ in os.walk(base):
        if "train" in dirs and "test" in dirs:
            return r
        if r.rstrip("/\\").count(os.sep) - base_depth >= 3:
            dirs[:] = []  # không đi sâu quá
    return None


def ensure_dataset(zip_path, data_dir):
    """
    Bảo đảm dữ liệu đã được giải nén ở đĩa cục bộ (đọc nhanh hơn Drive rất nhiều). Trả về thư mục gốc dữ liệu.
    Có file đánh dấu '.extracted_ok' để biết lần giải nén trước đã hoàn tất (tránh dùng dữ liệu dở dang).
    """
    import shutil
    import zipfile
    marker = os.path.join(data_dir, ".extracted_ok")
    if os.path.exists(marker):
        root = find_data_root(data_dir)
        if root:
            print(f"Đã có dữ liệu giải nén sẵn: {root}")
            return root
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Không thấy {zip_path}\n"
            'Lưu ý: nếu file nằm trong mục "Máy tính" (Computers) của Google Drive thì Colab KHÔNG đọc được. '
            'Hãy chép file zip vào "Drive của tôi" (My Drive) rồi sửa DRIVE_DIR / ZIP_NAME.')
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)  # xóa bản giải nén dở dang
    os.makedirs(data_dir, exist_ok=True)
    local_zip = data_dir.rstrip("/\\") + "_tmp.zip"
    t0 = time.time()
    print(f"Chép {os.path.getsize(zip_path) / 1e6:.0f} MB từ Drive về đĩa cục bộ ...")
    shutil.copyfile(zip_path, local_zip)
    print(f"  chép xong ({time.time() - t0:.0f}s). Đang giải nén ...")
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-o", local_zip, "-d", data_dir], check=True)
    else:
        with zipfile.ZipFile(local_zip) as z:
            z.extractall(data_dir)
    os.remove(local_zip)
    root = find_data_root(data_dir)
    if root is None:
        raise RuntimeError("Giải nén xong nhưng không thấy thư mục train/ và test/ — kiểm tra lại file zip.")
    open(marker, "w").close()
    print(f"Giải nén xong sau {time.time() - t0:.0f}s: {root}")
    return root


class StepTimer:
    """
    Đo thời gian một bước và ghi vào evidence/timing.csv (bằng chứng cho thầy).

    Cách dùng:
        with StepTimer("NB02 · Fine-tune OCR", evidence_dir):
            ...  # việc cần đo
    """

    def __init__(self, name, evidence_dir, filename="timing.csv"):
        self.name = name
        self.path = os.path.join(evidence_dir, filename)  # vòng 2 dùng file riêng: timing_v2.csv
        os.makedirs(evidence_dir, exist_ok=True)

    def __enter__(self):
        self.t0 = time.time()
        self.start = now_iso()
        print(f"▶ Bắt đầu: {self.name}  ({self.start})")
        return self

    def __exit__(self, exc_type, exc, tb):
        sec = time.time() - self.t0
        new_file = not os.path.exists(self.path)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["step", "start", "end", "seconds", "status"])
            w.writerow([self.name, self.start, now_iso(), round(sec, 1),
                        "ok" if exc_type is None else f"loi:{exc_type.__name__}"])
        icon = "✔" if exc_type is None else "✘"
        print(f"{icon} Kết thúc: {self.name} — mất {fmt_seconds(sec)} ({sec:.0f} giây)")
        return False  # không nuốt lỗi


def save_environment(evidence_dir, tag):
    """
    Lưu thông tin môi trường chạy (GPU, phiên bản thư viện...) vào
    evidence/environment_<tag>.json — để chứng minh kết quả chạy trên máy nào.
    """
    from importlib import metadata
    libs = ["torch", "torchvision", "numpy", "opencv-python", "opencv-python-headless", "Pillow",
            "pandas", "kornia", "albumentations", "scikit-image", "matplotlib"]
    versions = {}
    for lib in libs:
        try:
            versions[lib] = metadata.version(lib)
        except Exception:  # noqa: BLE001 - thư viện không cài thì bỏ qua
            pass
    info = {
        "ghi_luc": now_iso(),
        "python": platform.python_version(),
        "he_dieu_hanh": platform.platform(),
        "torch_cuda_kha_dung": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "phien_ban_thu_vien": versions,
    }
    try:  # nvidia-smi chỉ có khi máy có GPU
        info["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                                            timeout=20).stdout
    except Exception:  # noqa: BLE001
        info["nvidia_smi"] = None
    write_json(os.path.join(evidence_dir, f"environment_{tag}.json"), info)
    return info
