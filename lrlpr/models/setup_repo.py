# -*- coding: utf-8 -*-
"""
setup_repo.py — Chuẩn bị repo LPSRGAN trên Colab.

Việc làm:
  1. Tải repo https://github.com/valfride/lpsrgan về đúng commit đã dùng khi lập kế hoạch
     (để kết quả không đổi nếu tác giả cập nhật code sau này).
  2. Áp MỘT bản vá nhỏ lên repo (xem apply_patches) rồi ghi lại vào evidence/patches.log.
  3. Tải trọng số LPSRGAN đã công bố (nếu chưa có).

Tất cả các hàm đều "idempotent": chạy lại nhiều lần không gây hại.
"""

import os
import subprocess
import urllib.request

REPO_URL = "https://github.com/valfride/lpsrgan.git"
REPO_COMMIT = "68fc78aa12bd9becf6ab322b3dacd890fab928a1"
PRETRAINED_URL = ("https://github.com/valfride/lpsrgan/releases/download/"
                  "LPSRGAN_pre_trained_model/best_model_Epoch_116.pth")


def _log(msg, log_path):
    """In ra màn hình và ghi thêm vào file nhật ký vá lỗi."""
    print(msg)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def clone_repo(repo_dir, log_path=None):
    """Tải repo về thư mục repo_dir và chuyển sang đúng commit. Bỏ qua nếu đã có."""
    if os.path.isdir(os.path.join(repo_dir, ".git")):
        _log(f"[repo] đã có sẵn {repo_dir}, bỏ qua bước tải", log_path)
    else:
        subprocess.run(["git", "clone", "-q", REPO_URL, repo_dir], check=True)
        _log(f"[repo] đã tải {REPO_URL} -> {repo_dir}", log_path)
    subprocess.run(["git", "-C", repo_dir, "checkout", "-q", REPO_COMMIT], check=True)
    head = subprocess.run(["git", "-C", repo_dir, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    assert head == REPO_COMMIT, f"Sai commit: {head}"
    _log(f"[repo] đang ở commit {head}", log_path)


def _comment_out_line(repo_dir, rel_path, target, tag, why, log_path):
    """
    Vô hiệu hóa (đưa thành chú thích) dòng 'target' trong file rel_path của repo.
    Có kiểm tra (assert) để biết ngay nếu repo đã khác so với dự kiến.
    """
    path = os.path.join(repo_dir, rel_path)
    with open(path, "r", encoding="utf-8", newline="") as f:
        text = f.read()
    marker = "# (da bo) " + target
    if marker in text:
        _log(f"[vá {tag}] {rel_path}: đã vá từ trước, bỏ qua", log_path)
        return
    assert target in text, f"Không thấy dòng cần vá trong {rel_path} (repo đã đổi?)"
    new_text = text.replace(target, marker + "  # " + why, 1)
    assert marker in new_text
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(new_text)
    _log(f"[vá {tag}] {rel_path}: đã bỏ '{target}' ({why})", log_path)


def apply_patches(repo_dir, log_path=None):
    """
    Hai bản vá nhỏ, chỉ để repo import được trên Colab mà không cần thư viện thừa:

      P1  models/__init__.py : bỏ 'from . import ocr_rodosol'
          Lý do: file đó import keras/tensorflow (OCR Keras của tác giả) mà ta không dùng;
          giữ lại sẽ kéo theo TensorFlow và dễ xung đột phiên bản.
      P2  models/lpsrgan_arch.py : bỏ 'from basicsr.archs.rrdbnet_arch import RRDBNet'
          Lý do: thư viện basicsr rất nặng, hay lỗi khi cài, và RRDBNet KHÔNG được dùng
          ở bất kỳ đâu trong file (chỉ import cho có).
    """
    _comment_out_line(repo_dir, os.path.join("models", "__init__.py"),
                      "from . import ocr_rodosol", "P1", "tránh kéo theo TensorFlow", log_path)
    _comment_out_line(repo_dir, os.path.join("models", "lpsrgan_arch.py"),
                      "from basicsr.archs.rrdbnet_arch import RRDBNet", "P2",
                      "basicsr nặng và không được dùng", log_path)


def download(url, dest, log_path=None):
    """Tải một file về dest (bỏ qua nếu đã có)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        _log(f"[tải] đã có {dest}", log_path)
        return dest
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    _log(f"[tải] {url} -> {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)", log_path)
    return dest


def setup(repo_dir, evidence_dir=None):
    """Tải repo đúng commit + áp bản vá. Trả về repo_dir."""
    log_path = os.path.join(evidence_dir, "patches.log") if evidence_dir else None
    clone_repo(repo_dir, log_path)
    apply_patches(repo_dir, log_path)
    return repo_dir
