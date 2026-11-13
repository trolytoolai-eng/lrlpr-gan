# LRLPR: làm nét rồi đọc biển số từ ảnh độ phân giải thấp

Đồ án học phần Thị giác máy tính (IT4343E), Đại học Bách khoa Hà Nội.
Sinh viên: Đào Anh Quân (20261211M). Giảng viên hướng dẫn: ThS. Nguyễn Duy Tùng.

Bài toán: đọc biển số xe (7 ký tự) từ 5 ảnh mờ, rất nhỏ của cùng một xe, trên bộ dữ liệu LRLPR-26 (ICPR 2026).
Cách làm: làm nét ảnh bằng mạng GAN (LPSRGAN), rồi đọc bằng OCR (GP-LPR), rồi gộp kết quả của 5 ảnh.

Cấu trúc thư mục bám theo repo của tác giả LPSRGAN ([valfride/lpsrgan](https://github.com/valfride/lpsrgan)):
một thư mục `config/` chứa file yaml cho từng nhánh, một file huấn luyện `train.py`, một file đánh giá `evaluate.py`.

## Kết quả (tập test, 3.000 xe, Exact Match = đọc đúng cả 7 ký tự, bỏ phiếu 5 ảnh)

| Nhánh | Mô tả | Exact Match |
|---|---|---|
| M0 | Phóng to ảnh LR | 0,6% |
| M1 | Xử lý ảnh cổ điển | 0,6% |
| M2 | LPSRGAN công bố, chưa huấn luyện lại | 4,0% |
| M3 | Fine-tune (L1 + VGG19) | 4,9% |
| M4 | Fine-tune + tăng cường dữ liệu | 5,5% |
| M5c | Thêm loss OCR có gradient (w = 3,0) | 53,1% |
| M5c+ | M5c, OCR dạy lại trên ảnh GAN | 58,7% |
| M6a | Mạng nhìn cả 5 ảnh cùng lúc | 55,5% |
| **M6a+** | M6a, OCR dạy lại | **60,6%** |

Số liệu đầy đủ, khoảng tin cậy, phân tích và hạn chế nằm trong báo cáo.
Mọi con số được tính lại từ file dự đoán từng xe (`results/<nhánh>/pred_*.csv`), không gõ tay.

## Cấu trúc

```
config/
  default.yaml            đường dẫn, seed, số tiến trình nạp dữ liệu
  LRLPR/                  mỗi nhánh một file: m3, m4, m5a, m5b, m5c, m6a, ocr_ft, ocr_sr
lrlpr/
  datasets/               đọc track, tiền xử lý (đệm 3:1, CLAHE), tăng cường dữ liệu, Dataset
  models/                 nạp GP-LPR và LPSRGAN, mở rộng lớp đầu cho mạng nhiều frame, vá repo gốc
  losses/                 perceptual loss VGG19
  train_funcs/            hàm phụ cho huấn luyện (kiểm tra val, cosine lr, dựng cache OCR)
  utils/                  hằng số, tiện ích chung, đọc yaml
  utils_test/             đánh giá, gộp 5 frame, độ đo, báo cáo và biểu đồ
split.py                  tách dữ liệu train / val / test
train_ocr.py              fine-tune OCR GP-LPR trên ảnh HR
train.py                  fine-tune mạng làm nét (M3, M4, M5, M6), tham số lấy từ config/LRLPR/*.yaml
train_ocr_sr.py           dạy lại OCR trên ảnh do GAN tạo ra (nhánh có dấu +)
evaluate.py               đánh giá một nhánh trên val hoặc test
show_track.py             bật một ví dụ lên (ảnh các nhánh cạnh nhau, chữ đọc được)
run_all.py                chạy cả quy trình theo ba vòng
notebooks/LRLPR_colab.ipynb   notebook mỏng để chạy trên Google Colab
```

## Cách chạy trên Google Colab (GPU Tesla T4)

1. Đặt `LRLPR-26.zip` và `models/Rodosol.pth` (trọng số GP-LPR) vào thư mục `MyDrive/IT4343E/`
   (mục Drive của tôi, không phải mục Máy tính vì Colab không đọc được mục này).
2. Mở `notebooks/LRLPR_colab.ipynb`, chọn GPU T4, chạy lần lượt các ô. Hoặc trong một ô:

```bash
git clone https://github.com/trolytoolai-eng/lrlpr-gan /content/lrlpr-gan
cd /content/lrlpr-gan
pip install -r requirements.txt
python run_all.py --round all
```

Thời gian chạy trên T4 (đo lúc thực hiện): vòng 1 khoảng 43 phút, vòng 2 khoảng 102 phút, vòng 3 khoảng 27 phút.
Colab có thể ngắt giữa chừng, nên chạy từng vòng (`--round 1`, rồi `--round 2`, rồi `--round 3`). Kết quả lưu trên Drive nên vòng sau dùng lại được.

### Các bước và tham số hữu ích

```bash
python run_all.py --round 1 --stage m3,eval_m3        # chỉ vài bước của vòng 1
python run_all.py --round 2 --only m5c                # chỉ huấn luyện nhánh M5c
python run_all.py --round 1 --set epochs=1 --set workers=0   # chạy thử rất nhanh
python run_all.py --round 3 --set m6a.init_from=M5a   # ghi đè riêng một nhánh
```

| Vòng | Các bước (`--stage`) |
|---|---|
| 1 | split, baselines, ocr, m3, eval_m3, m4, eval_m4, choose, test, report, bundle |
| 2 | check, m5, eval_m5, ocr_sr, eval_plus, choose, test, report, bundle |
| 3 | check, m6, eval_m6, choose, test, report, bundle |

Tra một ví dụ (khi cần minh họa):

```bash
python show_track.py --results_dir /content/drive/MyDrive/IT4343E/results --track test/track_10081 --split test --best M6a --out vi_du.png
```

## Quy tắc đánh giá

- Chọn mô hình và cách gộp 5 ảnh chỉ dựa trên tập val (`choose`), rồi mới chạy test.
- Tập test chỉ chạy một lần mỗi vòng. Có file khóa `evidence/test_run.lock` (và `_v2`, `_v3`) ghi giờ chạy và lựa chọn đã chốt. Chạy lần hai bị từ chối.
- Trong `evidence/` có thời gian từng bước, thông tin môi trường, mã băm sha256 của các file.

## Điểm khác với repo gốc của tác giả

Khi đọc mã LPSRGAN (commit `68fc78a`) em thấy các điểm sau và xử lý trong repo này:

- Loss OCR của repo gốc chỉ đếm số biển đọc sai nên không tạo gradient. Ở đây dùng cross-entropy của OCR đóng băng, gradient chảy ngược về mạng làm nét (`train.py`, hàm `step_fn`).
- Thành phần adversarial của repo gốc bị chú thích, nên thực tế chỉ học bằng L1 và VGG. Ở đây giữ đúng hành vi đó (`adv_weight: 0`), có thể bật lại.
- Hàm tăng cường dữ liệu của repo gốc làm co ảnh HR khi chọn phép xoay. Ở đây viết hàm riêng, ảnh HR luôn là đáp án thật (`lrlpr/datasets/preprocess.py`, hàm `augment_pair`).
- Hàm nạp mô hình của repo gốc nuốt lỗi âm thầm. Ở đây nạp chặt và kiểm tra lại sau khi nạp (`lrlpr/models/loaders.py`).
- Repo gốc được tải về đúng commit và vá hai dòng import (TensorFlow, basicsr) lúc chạy, xem `lrlpr/models/setup_repo.py`. Mã của tác giả không được chép vào repo này.

## Lưu ý về kết quả trong báo cáo

- Mã trong repo này được tổ chức lại từ mã đã chạy ba vòng thí nghiệm trên Colab. Thân các hàm giữ nguyên (đã kiểm tra tự động bằng cây cú pháp), chỉ thêm đọc yaml và gom ba notebook thành một chỗ chạy.
- Ở vòng 1, tập con val theo dõi lúc huấn luyện lấy 500 track đầu tiên (toàn ảnh ban ngày Brazilian). Từ vòng 2 mã lấy ngẫu nhiên đủ bốn nhóm (`stratified_subset`). Mã hiện tại dùng cách thứ hai cho cả ba vòng, nên chạy lại vòng 1 có thể lệch nhẹ so với số trong báo cáo.
- Chưa chạy lại toàn bộ trên Colab sau khi tổ chức lại mã. Đã chạy thử cả ba vòng trên dữ liệu giả nhỏ để kiểm tra luồng chạy.
- Hạn chế đã biết: OCR trong loss và OCR dùng để đo cùng là GP-LPR; thiếu mốc M0+ (OCR dạy lại trên ảnh LR không làm nét); mỗi cấu hình chỉ chạy một hạt giống.

## Tài liệu tham khảo chính

- Pan, Tang, Tjahjadi, LPSRGAN, Neurocomputing 580, 2024.
- Nascimento và cộng sự, LCOFL, SIBGRAPI 2024, arXiv:2408.15103.
- Laroca và cộng sự, ICPR 2026 Competition on Low-Resolution License Plate Recognition, arXiv:2604.22506.
- Liu và cộng sự, GP-LPR, MMM 2024.
