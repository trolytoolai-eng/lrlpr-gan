# -*- coding: utf-8 -*-
"""
Hằng số dùng chung: kích thước ảnh, bảng chữ cái biển số, tham số OCR GP-LPR, tên 4 nhóm dữ liệu.
"""




ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 36 ký tự biển số (số + chữ hoa)
PLATE_LEN = 7                 # biển số Brazil/Mercosur đều có 7 ký tự
N_FRAMES = 5                  # mỗi track có 5 ảnh LR (và 5 ảnh HR ở tập train)
LR_H, LR_W = 16, 48           # kích thước ảnh LR đưa vào mạng làm nét (GAN)
SR_H, SR_W = 32, 96           # kích thước ảnh sau làm nét = kích thước đầu vào của OCR (gấp đôi LR)
ASPECT = 3.0                  # tỉ lệ rộng/cao chuẩn của biển số sau khi đệm
ASPECT_TOL = 0.15             # sai số cho phép quanh tỉ lệ chuẩn (giống repo)
PAD_COLOR = (127, 127, 127)   # màu xám dùng để đệm (giống repo)

# Tham số mặc định của mô hình OCR GP-LPR (lấy từ file cấu hình test của repo)
DEFAULT_GPLPR_ARGS = dict(alphabet=ALPHABET, nc=3, imgW=SR_W, imgH=SR_H, K=PLATE_LEN,
                          isSeqModel=True, head=2, inner=256, isl2Norm=True)

# Thứ tự 4 nhóm dữ liệu dùng khi báo cáo kết quả
GROUPS = ["Scenario-A/Brazilian", "Scenario-A/Mercosur",
          "Scenario-B/Brazilian", "Scenario-B/Mercosur"]
