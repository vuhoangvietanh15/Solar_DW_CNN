# Solar DW-CNN — dự án riêng cho dữ liệu quan sát Mặt Trời

Chỉ huấn luyện **DW_CNN 1D**, không chạy DAN/FCN/ablation và không thay đổi dự án ECG.
Dữ liệu gốc được đọc ở `../Sun_Observering_Interferometry/Signal_Processing`, không sửa hoặc ghi đè.

## 1. Những điều đã xác minh từ mã nguồn và dữ liệu

Đã đọc các README của `Dunglaibaibao`, `Sun_Observering_Interferometry`, `esp32_deployment`,
cùng DW_CNN/DWT gốc, GNU Radio flowgraphs, khối ghi CSV và các notebook xử lý tín hiệu.

| Thư mục | Mục đích |
|---|---|
| `../Dunglaibaibao` | ECG MIT-BIH/NSTDB, học có giám sát noisy/clean và so sánh nhiều mạng |
| `../esp32_deployment` | Xuất mô hình ECG INT8/TFLite và chạy ESP32; không dùng cho bài toán này |
| `../Sun_Observering_Interferometry` | Mã thu, CSV quan sát, notebook và hình kết quả; README rất ngắn |
| `Solar_DW_CNN` (thư mục này) | Chuẩn bị CSV, train DW_CNN, xuất dự đoán và kiểm tra biến dạng |

Có **10.189 CSV**, mỗi file không header, hai cột, **512 hàng**:

- Cột 0: nhãn tần số được ghi theo MHz.
- Cột 1: giá trị log công suất được ghi theo dB, không phải I/Q hay điện áp miền thời gian.
- Đây không phải 512 kênh đầu vào: mỗi vector có 512 điểm và một kênh giá trị đo.
- Không có nhãn `clean` đáng tin cậy, không có raw I/Q, không có dữ liệu pha/cross-visibility trong CSV.
- Không có cơ sở coi đây là băng thông 50 MHz hoặc công suất đã hiệu chuẩn dBm/SFU.

| Phiên | Số CSV | Điểm đầu–cuối trong cột MHz | Dùng mặc định |
|---|---:|---|---|
| 2024_08_28 | 2500 | 621.4000–622.5977 | Tách riêng |
| 2024_10_11 | 523 | 609.5000–610.4980 | Tách riêng |
| 2025_03_20 | 1433 | 608.8000–611.1953 | Train |
| 2025_03_21 | 1433 | 608.8000–611.1953 | Train |
| 2025_03_22 | 1433 | 608.8000–611.1953 | Train |
| 2025_03_23 | 1434 | 608.8000–611.1953 | Validation |
| 2025_03_24 | 1433 | 608.8000–611.1953 | Test |

Tổng mặc định **4299 train / 1434 validation / 1433 test**. Chia theo ngày trước khi học,
không trộn frame gần nhau vào cả train và test. Không resample ba thiết lập quan sát thành một lưới giả.
`dataset_audit.json` ghi kết quả đọc toàn bộ bảy phiên, kiểm tra shape, hữu hạn và lưới tần số.

### Điểm cần xác minh về FFT trước khi diễn giải khoa học

**`Interferometer.py` và `interferometer.grc` thiếu FFT.** Chuỗi đang thấy là:

```
I/Q → stream_to_vector → complex_to_mag_squared → moving_average → 10*log10 → integration → CSV
```

Nếu CSV được thu đúng bằng chuỗi này, các phần tử vector là công suất của vị trí mẫu thời gian,
**không phải các bin phổ**, dù khối ghi gán nhãn MHz. Không thể khôi phục phổ thật bằng cách lấy FFT
của các giá trị dB đã mất pha này.

`dlite.grc` có FFT thuận, Blackman–Harris window, `shift=True`, sau đó mới tính công suất.
CSV không ghi phiên bản flowgraph, nên hiện chưa xác định được nhánh nào tạo ra từng ngày đo.
Mọi metadata/biểu đồ mới đều ghi **frequency unverified**. Pipeline vẫn chạy được trên vector đo,
nhưng chưa thể dùng kết quả để khẳng định phát xạ theo tần số.

Nếu nguồn là nhánh thiếu FFT, cần thu lại raw I/Q hoặc sửa chuỗi thu:
`vector → window + complex FFT → fftshift → |FFT|² → tích phân công suất tuyến tính → log để hiển thị`.
Phải ghi thêm sample rate, center frequency, FFT size, window/normalization, integration time,
gain/AGC và phiên bản chương trình. Thư mục này không tự sửa thiết bị/flowgraph gốc.

## 2. Chọn cách học đúng với dữ liệu chưa có clean

Không lấy phổ làm mịn/median làm “ground truth sạch”, không tự gán một anten là clean,
không train autoencoder với đầu vào bằng chính target nguyên vẹn.

Dự án dùng **masked self-supervision lấy cảm hứng từ Noise2Self**:

1. Chuẩn hóa `z = (recorded_db - center_train) / scale_train` bằng **hai scalar cố định**, chỉ fit trên train.
   Không chuẩn hóa riêng từng frame: thống kê từ frame có thể làm rò target bị che và mất biến thiên mức nền.
   Lưu hai hệ số để hoàn nguyên chính xác đơn vị dB.
2. Với mỗi frame, chọn ngẫu nhiên một pha trong chu kỳ 8. Che khoảng 1/8 số bin.
3. Thay bin bị che bằng trung bình hai hàng xóm **không bị che**. Biên dùng hàng xóm phía trong, không wrap.
4. Đưa `[giá trị đã che, binary mask]` vào mạng, target là **giá trị đo ban đầu**.
   Kênh thứ hai là mask, **không phải Q**.
5. MSE chỉ tính ở bin bị che. Đổi mask mỗi epoch; validation dùng mask cố định.
6. Inference cũng che lần lượt đủ 8 pha và chỉ nhận đầu ra tại bin đang bị che.
   Không chạy một lượt trên dữ liệu nguyên vẹn: sẽ khác phân phối học và không còn blind prediction.

Lõi DW_CNN vẫn gồm hai cấp Haar DWT/IDWT, bốn conv-block chính, hai skip cộng và head.
Mỗi block chính có 5 Conv1D 64 filters, kernel 3; bottleneck/decoder có projection 128 filters.
Đầu ra một kênh tuyến tính với residual correction khởi tạo bằng 0 trên giá trị đã che.
Input `(512, 2)`, output `(512, 1)`, **334.017 tham số** ở cấu hình mặc định.
Không tái sử dụng checkpoint ECG vì phân phối dữ liệu và số kênh đầu vào đã thay đổi.

Haar dùng cộng/trừ even/odd, tránh tạo ma trận convolution chéo kênh cho một phép biến đổi cố định.
Mô hình có custom layer/loss được đăng ký serialization, lưu đầy đủ `.keras`.
Mixed precision chạy conv bằng float16, head/loss float32; Adam có gradient clipping.

### Giới hạn cần hiểu trước khi train lâu

Noise2Self dựa trên giả thiết nhiễu giữa các phần tử được che và phần tử quan sát độc lập có điều kiện,
còn tín hiệu có thể dự đoán từ ngữ cảnh. **Chưa xác minh giả thiết đó với bộ thu này**;
window FFT, tích phân, RFI, AGC và các khối integration có thể tạo tương quan.
Trong miền dB, kỳ vọng log công suất cũng không bằng log của kỳ vọng công suất tuyến tính.

- Học dự đoán các giá trị log đo được, không phải trực tiếp khôi phục flux vật lý.
- RFI tồn tại liên tục/có tương quan có thể vẫn được mạng giữ lại.
- Một vạch thiên văn hẹp và một vạch RFI hẹp có thể không phân biệt được bằng vector đơn.
- Đỉnh hẹp/thay đổi nhanh có thể bị làm mượt. Skip connections không bảo đảm bảo toàn diện tích/đỉnh/năng lượng.
- Không có clean nên không báo SNR gain/PSNR hay “đã bảo toàn 100% năng lượng”.
- Không có pha/cặp anten nên không đánh giá được coherence, pha giao thoa hoặc visibility.
- Mục tiêu loss nhỏ hay waterfall đẹp không chứng minh phát xạ Mặt Trời thật được giữ nguyên.

Muốn train **supervised RFI excision** có kiểm chứng, cần thêm cặp clean/noisy đã căn chỉnh,
RFI mask đáng tin cậy, hoặc dữ liệu mô phỏng được kiểm định kèm phép thử chèn tín hiệu.
Pipeline hiện tại là phương án train thực hiện được trên chính bộ dữ liệu đang có, không thay thế bước đó.

Nguồn phương pháp: [Noise2Self, ICML 2019](https://proceedings.mlr.press/v97/batson19a.html).
Thiết lập GPU theo [TensorFlow mixed precision](https://www.tensorflow.org/guide/mixed_precision).

## 3. Chạy ngay bằng GPU hiện tại

Đã xác minh môi trường `/home/vietanh/miniconda3/envs/anh/bin/python` có TensorFlow 2.15.1,
nhận NVIDIA RTX A2000 Laptop **4 GB**, và chạy được huấn luyện mixed precision thực tế.
Không cần cài lại/mất môi trường ECG để chạy:

```bash
cd /home/vietanh/Downloads/ECG/Solar_DW_CNN
conda activate anh

# Dữ liệu mặc định đã được chuẩn bị tại data/prepared.
# Train đầy đủ: tuyệt đối KHÔNG thêm --steps-per-epoch/--validation-steps từ smoke test.
CUDA_VISIBLE_DEVICES=0 python -m solar_dwcnn.train \
  --data data/prepared \
  --run runs/solar_v1 \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.0001 \
  --mixed-precision
```

Nếu shell chưa có `conda activate`, dùng trực tiếp:

```bash
CUDA_VISIBLE_DEVICES=0 /home/vietanh/miniconda3/envs/anh/bin/python -m solar_dwcnn.train \
  --run runs/solar_v1 --epochs 100 --batch-size 16 --mixed-precision
```

`solar_v1` phải là tên run mới, tránh ghi đè checkpoint. Đã có run thì đặt `solar_v2`, v.v.
Mặc định dừng sớm sau 15 epoch validation không cải thiện; giảm learning rate khi chững lại.
Chọn checkpoint bằng validation, không dùng test để chọn epoch/hyperparameter.
GPU memory growth bật sẵn; không thấy GPU sẽ báo lỗi, không âm thầm train CPU.
Nếu CUDA OOM, giảm batch xuống 8 hoặc 4 và dùng tên run mới.
Nếu gặp NaN, chạy lại với learning rate nhỏ hơn và bỏ `--mixed-precision`; không tiếp tục dùng checkpoint lỗi.
Không bật XLA toàn mạng mặc định. Tốc độ phụ thuộc dữ liệu, driver và trạng thái nhiệt GPU.

### Tạo môi trường tách biệt nếu cần

Không bắt buộc nếu dùng `anh` đã kiểm tra. Không uninstall/reinstall môi trường ECG đang hoạt động.

```bash
conda create -n solar_dwcnn python=3.11 -y
conda activate solar_dwcnn
python -m pip install -r requirements.txt
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Requirements là cấu hình cài mới đề xuất theo TensorFlow 2.15.1; chưa tạo/cài mới môi trường này trong lần chỉnh sửa.
Cảnh báo đăng ký CUDA factory/TensorRT không tự đồng nghĩa GPU bị lỗi; kiểm tra danh sách GPU và chạy train thật.

## 4. Chuẩn bị lại dữ liệu và kiểm thử

Toàn bộ lệnh dưới đây chạy từ `Solar_DW_CNN`.

```bash
python -m solar_dwcnn.prepare --audit-only
python -m unittest discover -s tests -v

# Chọn output mới nếu data/prepared đã tồn tại.
python -m solar_dwcnn.prepare --output data/prepared_v2 \
  --train-days 2025_03_20 2025_03_21 2025_03_22 \
  --val-days 2025_03_23 \
  --test-days 2025_03_24
```

Khi dùng `prepared_v2`, truyền `--data data/prepared_v2` vào train, infer và diagnose.
Loader kiểm tra file lỗi, NaN/Inf, chiều dài/lưới không khớp, ngày bị trùng và nội dung CSV trùng giữa các split.
Timestamp từ tên file được diễn giải UTC+07, gồm cả phần thập phân giây; lưu epoch time vào NPZ.
Metadata lưu danh sách ngày, số frame, SHA256 từng CSV, nguồn dữ liệu và thống kê train.
Không gộp hai phiên năm 2024 vào run này. Muốn học từng phiên đó cần một phân chia thời gian có khoảng cách
và đánh giá riêng; CLI hiện chia theo thư mục ngày, không tự tạo random split trong một ngày.

## 5. Xuất waterfall và phổ sau khi train

```bash
CUDA_VISIBLE_DEVICES=0 python -m solar_dwcnn.infer \
  --run runs/solar_v1 --data data/prepared --split test \
  --output outputs/solar_v1_test --batch-size 16

CUDA_VISIBLE_DEVICES=0 python -m solar_dwcnn.diagnose \
  --run runs/solar_v1 --data data/prepared \
  --output outputs/solar_v1_feature_probe --samples 16
```

Mỗi lần xuất dùng thư mục mới. Output gồm:

- `prediction.npz`: `input_db`, `prediction_db`, `removed_db`, `frequency_mhz`, `timestamps`, `files`.
- `report.json`: độ thay đổi RMS, dịch mức nền trung bình, masked prediction MSE và mốc nội suy hàng xóm.
  Các số này **không phải lỗi so với tín hiệu sạch**.
- `frames.csv`: tên file và thống kê từng frame; `provenance.json`: cấu hình của checkpoint.
- `<ngày>/waterfall.png`: input, dự đoán và residual; input/dự đoán dùng **cùng giới hạn màu**, không nội suy ảnh.
  Giới hạn màu theo percentile 1–99 chỉ để hiển thị; giá trị NPZ không bị clip.
- `<ngày>/spectra_and_trend.png`: phổ ba thời điểm và đường trung bình giá trị log theo thời gian thực.
  Đây không phải tổng năng lượng/flux tuyến tính.
- Feature probe: chèn đỉnh Gaussian rộng 0.5–16 bin, biên độ một training std theo đơn vị dB;
  đo `F(x + đỉnh) - F(x)` và lưu tỷ lệ đáp ứng đỉnh/diện tích cùng hình minh họa.
  Đây là thử độ nhạy, không phải mô phỏng RFI vật lý hay ground truth bão Mặt Trời.

Inference ngày khác cùng thiết lập thu:

```bash
python -m solar_dwcnn.infer --run runs/solar_v1 \
  --day 2025_03_24 --output outputs/solar_v1_day24
```

Lưới khác sẽ bị từ chối. Biểu đồ chia riêng từng ngày; trục thời gian waterfall là thứ tự frame có nhãn giờ,
khoảng trống giữa frame không được vẽ tỷ lệ. NPZ giữ thời gian đầy đủ để phân tích lại.

## 6. Cách xem chất lượng trước khi dùng kết quả

1. Đọc `training_report.json` và `history.csv`. Loss phải hữu hạn; run đầy đủ cần tốt hơn mốc nội suy
   trên cùng validation mask. Mốc này chỉ kiểm tra mạng có học thêm được gì, không phải so sánh nhiều model.
2. Nhìn residual: cấu trúc dải rộng hay hình burst rõ trong phần bị loại bỏ là dấu hiệu cần xem xét lại.
3. Đọc feature probe: đáp ứng khoảng 1 là giữ được đặc trưng chèn vào, khoảng 0 là xóa;
   không áp dụng ngưỡng “đẹp” chung cho mọi mục tiêu. Giữ đỉnh hẹp và xóa vạch RFI hẹp là hai yêu cầu có thể xung đột.
4. Giữ nguyên phổ gốc. Không dùng ảnh waterfall đã đổi contrast riêng để chứng minh tăng SNR.
5. Xác minh nguồn FFT và hiệu chuẩn trước khi kết luận về phổ vật lý.
6. Khi có nhãn sạch/RFI thật, xây dựng bài kiểm định độc lập về sai số biên độ, diện tích, nhiễu ngoài vùng phát xạ
   và khả năng giữ burst; không dùng dữ liệu test hiện tại để điều chỉnh liên tục.

## 7. Bố cục và trạng thái kiểm tra

```
Solar_DW_CNN/
├── README.md
├── requirements.txt
├── dataset_audit.json
├── solar_dwcnn/
│   ├── data.py       # đọc CSV, kiểm tra, normalization và mask
│   ├── prepare.py    # audit/prepare CLI
│   ├── model.py      # DW_CNN, Haar DWT/IDWT, masked MSE
│   ├── runtime.py    # GPU/memory growth/mixed precision
│   ├── train.py      # học, best checkpoint, early stopping, logs
│   ├── infer.py      # blind inference, NPZ, spectrum/waterfall/residual
│   └── diagnose.py   # kiểm tra đáp ứng đỉnh rộng/hẹp
├── tests/test_pipeline.py
├── data/prepared/    # dữ liệu đã chia, chỉ đọc từ dữ liệu gốc
├── runs/            # các lần chạy độc lập
└── outputs/         # kết quả theo run
```

Các run `gpu_smoke`/`gpu_verified` là chạy thử rất ngắn để kiểm tra kỹ thuật, **không phải mô hình đã train đạt chất lượng**.
Checkpoint tốt cho sử dụng thực tế cần được tạo bằng lệnh train đầy đủ và kiểm tra như trên.
Không có cam kết “phổ đẹp nhất” khi nguồn thu và ground truth chưa được xác minh.

## 8. Tài liệu ý tưởng, chỉ tiêu và sơ đồ khối

Bộ tài liệu chuẩn bị báo cáo ngày 21/10/2026 nằm trong [docs/README.md](docs/README.md): báo cáo kỹ thuật chi tiết cho beginner, sơ đồ Mermaid và bản chữ, bộ chỉ tiêu có phân loại trạng thái, kịch bản trình bày và câu hỏi dự kiến. Số liệu `solar_v1` dùng snapshot riêng có thời điểm, tách khỏi kết quả chạy thử trong `VERIFICATION.md`.
