# -*- coding: utf-8 -*-
"""
evaluate.py: đánh giá MỘT nhánh trên một tập (giống test_mv.py của repo LPSRGAN, nhưng cho một nhánh).

Ví dụ:
  # Nhánh dùng mạng làm nét: chạy mạng, lưu ảnh, đọc bằng OCR
  python evaluate.py --repo /content/lpsrgan --data_root /content/data/LRLPR-26 --split_dir <drive>/splits \
      --results_dir <drive>/results --branch M5c --split val --ckpt <drive>/ckpt/lpsrgan_M5c/best.pth \
      --ocr_ckpt <drive>/ckpt/gplpr_ft.pth

  # Nhánh có dấu +: đọc lại ảnh đã có của nhánh gốc bằng OCR khác
  python evaluate.py ... --branch M5c+ --src_branch M5c --split val --ocr_ckpt <drive>/ckpt/gplpr_sr.pth

Kết quả: <results_dir>/<nhánh>/pred_<split>.csv (từng track) và eval_info_<split>.json.
Lưu ý: đánh giá trên test là việc chỉ làm MỘT lần (xem run_all.py và khóa test_run*.lock).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrlpr.utils import api as C  # noqa: E402
from lrlpr.utils_test import eval_lib as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="thư mục repo LPSRGAN đã vá")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--split_dir", required=True)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--branch", required=True, help="tên nhánh, ví dụ M5c hoặc M5c+")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--ckpt", default="", help="trọng số mạng làm nét; bỏ trống với M0, M1, HR hoặc nhánh có dấu +")
    ap.add_argument("--src_branch", default="", help="với nhánh dấu +: nhánh gốc có ảnh để đọc lại")
    ap.add_argument("--ocr_ckpt", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = C.index_tracks(C.read_split_csv(os.path.join(args.split_dir, f"split_{args.split}.csv")), args.data_root)
    ocr, _ = C.load_gplpr(args.repo, args.ocr_ckpt, device)
    if args.src_branch:
        E.evaluate_with_ocr(args.branch, args.src_branch, args.split, rows, args.results_dir, ocr, device, args.ocr_ckpt)
    else:
        E.evaluate_branch(args.branch, args.split, rows, args.results_dir, ocr, device, args.repo, args.ckpt or None)


if __name__ == "__main__":
    main()
