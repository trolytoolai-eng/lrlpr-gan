# -*- coding: utf-8 -*-
"""
show_track.py: bật MỘT ví dụ lên (để trả lời khi giảng viên hỏi). Chỉ đọc lại file đã lưu, không cần GPU.

Mỗi cột là một nhánh, 5 hàng là 5 frame, chữ nhỏ bên dưới là biển số OCR đọc được (xanh = đúng, đỏ = sai).

Ví dụ:
  python show_track.py --results_dir <drive>/results --track test/track_10081 --split test --best M6a --out ví_dụ.png
Mã track lấy ở cột track của file pred_<split>.csv. Bỏ --track để liệt kê các ca tiêu biểu.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lrlpr.utils_test import report_lib as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--track", default="", help="ví dụ test/track_10081; bỏ trống để liệt kê ca tiêu biểu")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--fusion", default="mvcp_5", help="cách gộp 5 frame (mặc định: bỏ phiếu ký tự)")
    ap.add_argument("--best", default="", help="nhánh tốt nhất, luôn được vẽ trong hình (ví dụ M6a)")
    ap.add_argument("--out", default="", help="lưu hình ra file png (bỏ trống = chỉ hiện)")
    args = ap.parse_args()

    store = R.ResultStore(args.results_dir)
    best = args.best or None
    if args.track:
        R.show_track(store, args.track, args.split, args.fusion, best=best, out_png=args.out or None, show=not args.out)
    else:
        for cat, tracks in R.pick_examples(store, args.split, args.fusion, best).items():
            print(cat, ":", tracks)


if __name__ == "__main__":
    main()
