# LRLPR: làm nét bằng GAN rồi đọc biển số

Đồ án học phần Thị giác máy tính (IT4343E), Đại học Bách khoa Hà Nội.
Sinh viên: Đào Anh Quân (20261211M). Giảng viên hướng dẫn: ThS. Nguyễn Duy Tùng.
English: [README.md](README.md)

**Bài toán.** Đọc biển số 7 ký tự từ 5 ảnh rất nhỏ và mờ của cùng một xe (bộ dữ liệu LRLPR-26, cuộc thi ICPR 2026).

**Cách làm.** Làm nét từng ảnh bằng GAN (LPSRGAN), đọc bằng OCR (GP-LPR), rồi bỏ phiếu 5 lần đọc.
Ý tưởng chính: cho OCR đã cố định chấm bài mạng làm nét lúc huấn luyện (loss OCR có gradient thật).

## Kết quả (tập test, 3.000 xe, Exact Match = đọc đúng cả 7 ký tự, bỏ phiếu 5 ảnh)

| Nhánh | Mô tả | Exact Match |
|---|---|---|
| M0 | Phóng to ảnh LR | 0,6% |
| M1 | Xử lý ảnh cổ điển | 0,6% |
| M2 | LPSRGAN công bố, chưa học thêm | 4,0% |
| M3 | Fine-tune (L1 + VGG19) | 4,9% |
| M4 | Fine-tune + tăng cường dữ liệu | 5,5% |
| M5c | Thêm loss OCR (w = 3,0) | 53,1% |
| M5c+ | M5c, OCR dạy lại trên ảnh GAN | 58,7% |
| M6a | Mạng nhìn cả 5 ảnh cùng lúc | 55,5% |
| **M6a+** | M6a, OCR dạy lại | **60,6%** (khoảng tin cậy 95%: 58,9% đến 62,4%) |

Bảng đầy đủ, khoảng tin cậy, kiểm định McNemar, phân tích và hạn chế nằm trong báo cáo.
Mọi con số trong báo cáo được tính lại từ file dự đoán từng xe, không gõ tay.

## Cấu trúc repo

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
notebooks/
  LRLPR_colab.ipynb       notebook gọn chạy mã đã sắp xếp lại trên Google Colab
  run_history/            ba notebook đã chạy thật (còn nguyên kết quả chạy), xem bên dưới
```

## Các notebook

Ba notebook trong `notebooks/run_history/` là các lần chạy Colab thật tạo ra số liệu trong báo cáo (Tesla T4, ngày 13/11/2026).
Kết quả chạy (log, bảng, hình) được giữ lại để kiểm tra lịch sử chạy.
Chúng dùng mã **trước khi** sắp xếp lại (một module phẳng) và thư mục Drive `MyCV`.
Mã trong repo này là cùng một mã được tổ chức gọn hơn.

| Notebook | Vòng | Làm gì | Thời gian trên T4 |
|---|---|---|---|
| `LRLPR_Round1_full_pipeline.ipynb` | 1 | Tách dữ liệu, mốc M0 đến M2, fine-tune OCR, M3, M4, chọn trên val, test một lần | khoảng 43 phút |
| `LRLPR_Round2_ocr_loss_ocr_retrain.ipynb` | 2 | Loss OCR (M5a, M5b, M5c), dạy lại OCR (các nhánh +), test một lần | khoảng 102 phút |
| `LRLPR_Round3_multiframe.ipynb` | 3 | Mạng nhìn 5 ảnh (M6a), test một lần | khoảng 27 phút |

## Bộ dữ liệu KHÔNG nằm trong repo

LRLPR-26 do các tác giả phát hành chỉ cho nghiên cứu học thuật, phi thương mại. Điều khoản ghi rõ: phân phối lại, toàn bộ hay một phần,
khi chưa có sự cho phép của tác giả là không được phép, và ảnh chỉ được dùng để minh họa trong ấn phẩm và bài thuyết trình học thuật.
Vì vậy repo này **không chứa file dữ liệu nào** (không có zip, không có ảnh, không có nhãn, không có file dự đoán kèm đáp án).

Muốn có dữ liệu, làm theo hướng dẫn ở trang chính thức:
<https://github.com/raysonlaroca/lrlpr-26-dataset> (ký thỏa thuận và gửi từ thư điện tử của trường đại học).

Sau khi có `LRLPR-26.zip`, kiểm tra xem có đúng file đã dùng cho báo cáo không:

```
dung lượng: 738.164.264 byte
sha256    : 4c2177880223d280beb6c6fd114d2008a47976e2fd6eb7adb0a62657182c5395
nội dung  : train/Scenario-A/{Brazilian,Mercosur}/track_*  (5 ảnh LR + 5 ảnh HR + annotations.json)
            train/Scenario-B/{Brazilian,Mercosur}/track_*
            test/track_*                                   (5 ảnh LR + annotations.json)
            20.000 track huấn luyện và 3.000 track test
```

Cần thêm trọng số OCR GP-LPR `Rodosol.pth` (do tác giả GP-LPR phát hành, không có trong repo).

## Cách chạy trên Google Colab (GPU Tesla T4)

1. Đặt `LRLPR-26.zip` và `models/Rodosol.pth` vào `MyDrive/IT4343E/` (mục Drive của tôi, không phải mục Máy tính vì Colab không đọc được mục đó).
2. Mở `notebooks/LRLPR_colab.ipynb`, chọn GPU T4 và chạy các ô. Hoặc trong một ô:

```bash
git clone https://github.com/trolytoolai-eng/lrlpr-gan /content/lrlpr-gan
cd /content/lrlpr-gan
pip install -r requirements.txt
python run_all.py --round all
```

Colab có thể ngắt giữa chừng nên an toàn hơn là chạy từng vòng (`--round 1`, rồi `--round 2`, rồi `--round 3`).
Kết quả lưu trên Drive nên vòng sau dùng lại được vòng trước.

Các tham số hữu ích và bảng các bước của từng vòng giống bản tiếng Anh ([README.md](README.md)).

## Quy tắc đánh giá

- Chọn mô hình và cách gộp 5 ảnh chỉ dựa trên tập val (`choose`), rồi mới chạy test.
- Tập test chỉ chạy một lần mỗi vòng. File khóa `evidence/test_run.lock` (và `_v2`, `_v3`) ghi giờ chạy và lựa chọn đã chốt. Chạy lần hai bị từ chối.
- Thư mục `evidence/` còn có thời gian từng bước, thông tin môi trường, mã băm sha256 của các file.

## Điểm khác với mã LPSRGAN gốc

Khi đọc mã LPSRGAN (commit `68fc78a`) nhóm thấy các điểm sau và xử lý trong repo này:

- Loss OCR của mã gốc chỉ đếm số biển đọc sai nên không tạo gradient. Ở đây dùng cross-entropy của OCR đóng băng, gradient chảy ngược về mạng làm nét (`train.py`, hàm `step_fn`).
- Thành phần adversarial của mã gốc bị chú thích, nên thực tế chỉ học bằng L1 và VGG. Ở đây giữ đúng hành vi đó (`adv_weight: 0`), có thể bật lại.
- Hàm tăng cường dữ liệu của mã gốc làm co ảnh HR khi chọn phép xoay. Nhóm viết hàm riêng, ảnh HR luôn là đáp án thật (`lrlpr/datasets/preprocess.py`, `augment_pair`).
- Hàm nạp mô hình của mã gốc nuốt lỗi âm thầm. Ở đây nạp chặt và kiểm tra lại sau khi nạp (`lrlpr/models/loaders.py`).
- Mã gốc được tải về đúng commit và vá hai dòng import (TensorFlow, basicsr) lúc chạy, xem `lrlpr/models/setup_repo.py`. Mã của tác giả không được chép vào repo này.

## Lưu ý về kết quả

- Mã trong repo được tổ chức lại từ mã đã chạy ba vòng trên Colab. Thân các hàm giữ nguyên (kiểm tra tự động bằng cây cú pháp), chỉ thêm đọc yaml và gom ba notebook thành một chỗ chạy.
- Ở vòng 1, tập con val theo dõi lúc huấn luyện lấy 500 track đầu tiên (toàn ảnh ban ngày Brazilian). Từ vòng 2 mã lấy ngẫu nhiên đủ bốn nhóm (`stratified_subset`). Mã hiện tại dùng cách thứ hai cho cả ba vòng nên chạy lại vòng 1 có thể lệch nhẹ so với báo cáo.
- Chưa chạy lại toàn bộ trên Colab sau khi tổ chức lại mã. Đã chạy thử cả ba vòng trên dữ liệu giả nhỏ để kiểm tra luồng chạy.
- Hạn chế đã biết: OCR trong loss và OCR dùng để chấm là cùng một OCR (GP-LPR); thiếu mốc M0+ (OCR dạy lại trên ảnh LR chưa làm nét); mỗi cấu hình chỉ chạy một hạt giống.
