# Kiểm tra đã thực hiện

Ngày 20/10/2026, môi trường `anh`, TensorFlow 2.15.1, RTX A2000 Laptop 4 GB.

- Đọc/kiểm tra 10.189 CSV thuộc 7 phiên; lưu `dataset_audit.json`.
- Chuẩn bị 4299/1434/1433 frame train/validation/test, không sửa dữ liệu gốc.
- `python -m unittest discover -s tests -v`: **5 tests passed**, xem `test_results.log`.
  Bao gồm round-trip/energy/gradient Haar, normalization chỉ fit train, từ chối split/lưới sai,
  target không rò qua mask kể cả biên, masked loss, lưu/nạp model và blind inference.
- `gpu_verified.log`: model 334.017 parameters, batch 16, mixed_float16, 2 epoch × 3 train steps,
  validation toàn bộ ngày 23/03. Đây là smoke test, chưa hội tụ.
- `runs/gpu_verified/best.keras`: checkpoint được lưu thành công.
- `inference_smoke.log`: nạp checkpoint mixed precision và xuất toàn bộ 1433 test frame.
  Đã mở xem ảnh `outputs/gpu_smoke_test/2025_03_24/waterfall.png` để kiểm tra hiển thị.
- `feature_probe.log`: nạp checkpoint `gpu_verified` và thử 6 độ rộng đỉnh trên 4 frame test;
  kết quả tại `outputs/gpu_verified_probe`. Probe chạy trên hàm inference cuối cùng đã tối ưu bằng tf.function.
- `gpu_smoke` là lần thử đầu dùng validation rút gọn; không dùng báo cáo của nó để so với baseline toàn validation.
- Không chạy đủ 100 epoch, không đánh giá SNR sạch, không xác nhận FFT nguồn thu hoặc bảo toàn flux.
- Chưa cài/thử môi trường mới từ requirements; kiểm tra GPU dùng môi trường `anh` sẵn có.

Checkpoint và hình chạy thử chỉ xác nhận pipeline hoạt động, không phải kết quả khử nhiễu đã đạt chuẩn khoa học.
