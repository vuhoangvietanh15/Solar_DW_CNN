# Kịch bản báo cáo và câu hỏi dự kiến

**Dùng cùng:** [Báo cáo kỹ thuật đầy đủ](BAO_CAO_Y_TUONG_CHI_TIEU_SO_DO_KHOI.md).  
**Chuẩn bị cho:** 21/10/2026.  
**Lưu ý:** số liệu dưới đây thuộc snapshot 10:15:27 ngày 20/10/2026, UTC+07. Cập nhật kết quả cuối nếu đã có, nhưng phải ghi lại nguồn và thời điểm mới.

## 1. Bản trình bày khoảng 5–7 phút

### Slide 1 — Bài toán và mục tiêu

**Nội dung slide:**

- Nghiên cứu DW-CNN 1D cho vector đo trong quan sát vô tuyến Mặt Trời.
- Giảm thành phần khó dự đoán, đồng thời kiểm tra việc giữ đặc trưng.
- Phổ dễ nhìn cần đi kèm đánh giá biến dạng và xác minh nguồn thu.

**Lời nói gợi ý:**

“Dự án chuyển kiến trúc DW-CNN từ ECG sang dữ liệu quan sát Mặt Trời. Mục tiêu là nghiên cứu giảm nhiễu trên vector đo, nhưng không chỉ làm ảnh mịn: em còn kiểm tra mức nền và khả năng giữ các đỉnh. Do nguồn FFT chưa được xác minh, em giới hạn kết luận ở dữ liệu ghi nhận hiện có.”

### Slide 2 — Dữ liệu thật

**Nội dung slide:**

- 10.189 CSV thuộc 7 ngày; mỗi CSV 512 điểm, nhãn MHz và giá trị dB.
- Dùng nhóm tháng 3/2025 cùng lưới: 4.299 train / 1.434 validation / 1.433 test.
- Chưa có clean, nhãn RFI hoặc I/Q gốc trong CSV sử dụng.

**Lời nói gợi ý:**

“Em chia dữ liệu theo ngày để giảm nguy cơ frame gần nhau rơi vào cả train và test. Hai ngày năm 2024 có lưới khác nên không ghép vào run hiện tại. Các CSV không cung cấp tín hiệu sạch, do đó không áp dụng nguyên cách học có giám sát của ECG.”

### Slide 3 — Ý tưởng tự giám sát

**Nội dung slide:**

- Chuẩn hóa bằng mean/std từ train.
- Che 64/512 điểm mỗi frame.
- Input: giá trị đã che + mask; loss chỉ tính điểm bị che.
- Inference đủ 8 pha để có dự đoán cho cả vector.

**Lời nói gợi ý:**

“Em tạo bài toán đoán điểm bị che từ ngữ cảnh. Target vẫn là số đo ban đầu, không gọi là dữ liệu sạch. Nếu đưa nguyên input làm target thì mạng có thể chỉ học sao chép. Cơ chế che tránh điều đó, nhưng hiệu quả khử nhiễu còn phụ thuộc giả thiết nhiễu và cấu trúc tín hiệu.”

### Slide 4 — Sơ đồ mạng

**Nội dung slide:** dùng sơ đồ tại mục 5.3 của báo cáo đầy đủ.

- Encoder–decoder hai cấp Haar.
- 25 Conv1D, 64 filters cơ sở, 334.017 tham số.
- Hai skip cộng và head residual trên giá trị đã che.

**Lời nói gợi ý:**

“Haar tách các đặc trưng tổng hợp và chênh lệch, sau đó IDWT khôi phục chiều dài. Hai skip nối encoder sang decoder. Đây là cơ chế biểu diễn đa mức; không được hiểu rằng hệ số detail chắc chắn là RFI hay skip tự bảo toàn năng lượng của tín hiệu thật.”

### Slide 5 — Bộ chỉ tiêu

**Nội dung slide:**

| Nhóm | Chỉ tiêu chính |
|---|---|
| Phần mềm/dữ liệu | Kiểm tra lưới, split, normalization, mask và lưu/nạp |
| Khả năng học | Masked MSE validation so với mốc nội suy |
| Tác động đầu ra | RMS thay đổi, dịch mean, residual cùng thang màu |
| Giữ cấu trúc | Đáp ứng đỉnh và hình dạng khi chèn Gaussian nhiều độ rộng |
| Chưa đủ điều kiện đo | SNR sạch, RFI F1, flux, pha giao thoa |

**Lời nói gợi ý:**

“Em phân biệt chỉ tiêu đánh giá việc dự đoán và chỉ tiêu đánh giá biến dạng. Ví dụ RMS thay đổi lớn không tự chứng minh đã lọc tốt. Với bộ dữ liệu hiện có, em chưa báo cáo SNR sạch hay tỷ lệ phát hiện RFI vì thiếu tham chiếu phù hợp.”

### Slide 6 — Kết quả và trạng thái

**Nội dung slide:**

- 5 test methods đã qua; GPU RTX A2000 4 GB chạy được mixed precision, batch 16.
- Snapshot `solar_v1`: đã ghi 29 epoch; best validation masked MSE 0,929566.
- Mốc nội suy 1,269735; giảm masked MSE khoảng 26,79%.
- Snapshot chưa có báo cáo kết thúc run.

**Lời nói gợi ý:**

“Tại thời điểm chụp số liệu, mô hình dự đoán điểm che tốt hơn mốc nội suy khoảng 26,79% về masked MSE. Đây chưa phải mức tăng SNR hay phần trăm RFI bị loại. Run hiện tại cần chốt kết thúc và đánh giá đúng checkpoint trước khi trình bày kết quả cuối.”

### Slide 7 — Rủi ro và bước tiếp theo

**Nội dung slide:**

- Xác minh nhánh FFT tạo dữ liệu.
- Đỉnh hẹp thật có thể bị giảm giống RFI hẹp.
- Chạy feature probe trên checkpoint cuối; kiểm tra residual.
- Bổ sung dữ liệu tham chiếu, nhiều ngày độc lập và metadata thu.

**Lời nói gợi ý:**

“Phép thử trên checkpoint chạy thử đã chỉ ra rủi ro làm giảm đỉnh hẹp. Vì vậy em chưa khẳng định mô hình giữ mọi vi cấu trúc hay loại toàn bộ RFI. Đóng góp hiện tại là pipeline độc lập, có kiểm tra, có lưu vết và có công cụ đánh giá hạn chế đó.”

## 2. Bản nói ngắn khoảng một phút

“Dự án của em ứng dụng DW-CNN 1D để nghiên cứu giảm nhiễu trên các vector đo quan sát Mặt Trời. Em đã kiểm tra 10.189 CSV và chọn nhóm tháng 3/2025 cùng lưới để chia theo ngày thành train, validation và test. Vì chưa có target sạch, em dùng học tự giám sát: che một phần dữ liệu, dự đoán từ ngữ cảnh và chỉ tính loss tại điểm che. Mạng có hai cấp Haar, 25 Conv1D và 334.017 tham số. Pipeline đã qua kiểm thử và train được trên GPU. Snapshot hiện tại có masked MSE validation thấp hơn mốc nội suy khoảng 26,79%, nhưng con số này chưa phải mức tăng SNR hay tỷ lệ loại RFI. Em đang kiểm tra khả năng giữ đỉnh và cần xác minh nguồn FFT trước khi kết luận về phổ vật lý.”

## 3. Câu hỏi có thể được hỏi

### 1. Không có dữ liệu sạch thì học kiểu gì?

Che một nhóm điểm để mạng không nhìn thấy chúng, dự đoán từ ngữ cảnh và so với giá trị đo bị che. Cách này tạo mục tiêu học từ dữ liệu đang có. Khả năng diễn giải thành khử nhiễu phụ thuộc giả thiết thống kê; không tự biến số đo thành ground truth sạch.

### 2. Tại sao input có hai kênh? Có phải I và Q không?

Không. Một kênh là giá trị đã che, kênh kia là mask 0/1. Dữ liệu CSV hiện không có I/Q phức.

### 3. Mạng có chắc phân biệt được burst với RFI không?

Chưa chắc. Hai hiện tượng có thể giống nhau trong một vector, đặc biệt ở vạch rất hẹp. Mạng hiện không có nhãn RFI và không dùng ngữ cảnh thời gian giữa frame.

### 4. Wavelet có tự lọc nhiễu không?

Không. DWT là phép phân rã. Việc giữ/bỏ đặc trưng là do mạng học trong điều kiện dữ liệu và loss; hệ số H lớn không đồng nghĩa là nhiễu.

### 5. Có giữ nguyên năng lượng nhờ Haar và skip không?

Không thể khẳng định như vậy cho toàn mạng. Cặp Haar thuần có tính chất tái tạo/bảo toàn tổng bình phương trong số học lý tưởng; Conv/ReLU và đầu ra học được không bảo toàn năng lượng theo tính chất đó. Dữ liệu dB cũng chưa đủ để suy ra năng lượng vật lý đã hiệu chuẩn.

### 6. Vì sao cần chạy 8 lượt lúc inference?

Một pha chỉ che 64 điểm. Lần lượt che 8 nhóm và lấy output đúng vị trí bị che để mỗi điểm đều được dự đoán mà không nhìn thấy chính giá trị của nó.

### 7. 26,79% nghĩa là gì?

Mức giảm tương đối của masked MSE validation so với mốc nội suy hàng xóm ở snapshot đã lưu. Không phải 26,79% RFI bị loại, không phải độ chính xác 73,21% và không phải SNR tăng tương ứng.

### 8. Mốc nội suy có phải lại so sánh nhiều model không?

Không có thí nghiệm DAN/FCN/ablation trong dự án mới. Mốc nội suy chỉ là cách lấp điểm đơn giản đã dùng ở input; nó giúp kiểm tra mạng có học thêm được gì so với điểm khởi đầu.

### 9. Phổ nhìn mịn là thành công chưa?

Chưa. Phải xem residual, mức nền và đáp ứng đặc trưng. Xóa cả tín hiệu thật cũng có thể làm ảnh trông mịn.

### 10. Tại sao nói nhãn MHz chưa xác minh?

Nhánh `Interferometer` đang thấy thiếu FFT, còn `dlite.grc` có FFT. Chưa biết nhánh nào tạo từng CSV. Nếu số liệu đến từ nhánh thiếu FFT thì gán nhãn MHz không biến công suất mẫu thời gian thành phổ.

### 11. Có chạy realtime được không?

Chưa có benchmark end-to-end nên chưa kết luận. Đã train và infer trên GPU là bằng chứng vận hành phần mềm, chưa phải bằng chứng đạt độ trễ/tốc độ của hệ thu thực tế.

### 12. Thử đỉnh chèn có thay thế clean được không?

Không. Nó đo độ nhạy với một perturbation biết trước. Hiện phép chèn thực hiện trong miền log, chưa phải mô hình nguồn phát/RFI vật lý được hiệu chuẩn.

### 13. Checkpoint chạy thử và solar_v1 khác gì?

Checkpoint chạy thử học vài bước để kiểm tra kỹ thuật. `solar_v1` dùng toàn train/validation mỗi epoch. Không trộn loss của `solar_v1` với hình hoặc feature probe từ checkpoint thử để tạo một kết quả chung.

### 14. Chia ngày đã bảo đảm tổng quát hóa chưa?

Mới giảm một nguy cơ leakage. Chỉ một ngày test chưa đại diện đủ mọi điều kiện; vẫn cần nhiều phiên độc lập.

## 4. Những câu nên sửa trước khi nói

| Tránh nói | Cách diễn đạt có cơ sở |
|---|---|
| “Lọc sạch hoàn toàn RFI” | “Đang đánh giá khả năng giảm nhiễu; chưa có nhãn RFI để kết luận loại bỏ hoàn toàn.” |
| “Haar tách L là tín hiệu thật, H là nhiễu” | “L/H là hai nhóm hệ số đa mức; mỗi nhóm đều có thể chứa tín hiệu và nhiễu.” |
| “Đã bảo toàn 100% năng lượng” | “Đã có công cụ kiểm tra biến dạng, chưa chứng minh bảo toàn năng lượng vật lý.” |
| “Dữ liệu 610 MHz băng thông 50 MHz” | “Nhãn nhóm hiện dùng từ 608,8 đến 611,1953 MHz; nguồn FFT chưa xác minh.” |
| “Hai input channels là I/Q” | “Hai input channels là giá trị đã che và mask.” |
| “Tăng SNR 26,79%” | “Giảm masked MSE validation khoảng 26,79% so với mốc nội suy ở snapshot.” |
| “Mạng đã chứng minh giữ vi cấu trúc” | “Cần đo lại feature probe và thẩm định checkpoint cuối.” |

## 5. Chuẩn bị trước khi trình bày

1. Mở báo cáo đầy đủ, thử hiển thị Mermaid. Nếu trình xem không hỗ trợ, dùng sơ đồ chữ mục 5.6.
2. Chốt nguồn cho mỗi hình/số liệu: tên run, checkpoint, thời điểm snapshot và tập dữ liệu.
3. Nếu run đã kết thúc, đọc `training_report.json`, chụp snapshot mới và cập nhật đồng bộ slide; không sửa nhầm snapshot cũ như thể cùng thời điểm.
4. Nếu đã xuất hình mới, ghi rõ hình nào thuộc `solar_v1`. Hình chạy thử chỉ minh họa pipeline.
5. Nắm ba câu cốt lõi: thiếu clean → che điểm; low loss → dự đoán tốt hơn theo chỉ tiêu hiện có; đúng về phổ → cần xác minh chuỗi FFT và dữ liệu đo.
6. Khi chưa có số đo, trả lời “chưa đo, kế hoạch đo là…” và nêu điều kiện cần bổ sung.
