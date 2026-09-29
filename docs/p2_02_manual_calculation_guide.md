# Hướng dẫn tính tay và kiểm tra kết quả F2 Chunk-Level (P2-02)

Tài liệu này cung cấp công thức toán học chi tiết, các bước suy luận từng bước (step-by-step) để bất kỳ ai cũng có thể tự tính nhẩm/tính tay bằng bút giấy nhằm đối chiếu và xác minh kết quả đánh giá chunk-level metrics ($F_2$, Precision, Recall, $F_1$) của deliverable `evaluate_f2.py`.

---

## 1. Cơ sở toán học & Công thức chuẩn

Cho một query $q$:
- $T$: Tập hợp các chunk ID ground truth (thực tế liên quan). $|T|$ là số chunk thực tế.
- $P$: Tập hợp các chunk ID predicted (dự đoán liên quan). $|P|$ là số chunk dự đoán.
- $TP = |T \cap P|$: True Positives (số chunk đúng).
- $FP = |P \setminus T|$: False Positives (số chunk dự đoán thừa/sai).
- $FN = |T \setminus P|$: False Negatives (số chunk thực tế bị bỏ sót).

### Các metric per-query:
1. **Precision**: Tỷ lệ chunk đoán đúng trên tổng số chunk được dự đoán.
   $$\text{Precision} = \frac{TP}{|P|} \quad (\text{nếu } |P| = 0 \implies 0.0)$$

2. **Recall**: Tỷ lệ chunk tìm được trên tổng số chunk thực tế.
   $$\text{Recall} = \frac{TP}{|T|} \quad (\text{nếu } |T| = 0 \implies 0.0)$$

3. **$F_1$ Score**: Trung bình điều hòa cân bằng giữa Precision và Recall ($\beta=1$).
   $$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{|T| + |P|}$$

4. **$F_2$ Score**: Trung bình điều hòa trọng số gấp đôi cho Recall ($\beta=2$).
   Theo định nghĩa $F_\beta$:
   $$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$
   Với $\beta = 2$:
   $$F_2 = (1 + 4) \cdot \frac{\frac{TP}{|P|} \cdot \frac{TP}{|T|}}{4 \cdot \frac{TP}{|P|} + \frac{TP}{|T|}} = 5 \cdot \frac{\frac{TP^2}{|P| \cdot |T|}}{\frac{4 \cdot TP \cdot |T| + TP \cdot |P|}{|P| \cdot |T|}} = \frac{5 \cdot TP}{4 \cdot |T| + |P|}$$

   *(Quy ước: Khi $4 \cdot |T| + |P| = 0 \implies F_2 = 0.0$)*

### Macro Aggregation (Trung bình Macro trên toàn bộ $N$ queries):
$$\text{Macro Precision} = \frac{1}{N} \sum_{i=1}^N \text{Precision}_i$$
$$\text{Macro Recall} = \frac{1}{N} \sum_{i=1}^N \text{Recall}_i$$
$$\text{Macro } F_1 = \frac{1}{N} \sum_{i=1}^N F_{1, i}$$
$$\text{Macro } F_2 = \frac{1}{N} \sum_{i=1}^N F_{2, i}$$

---

## 2. Tính tay trên ví dụ nhỏ mẫu (Toy Reference)

Tệp phụ: [`tests/fixtures/toy_chunk_f2_reference.csv`](../tests/fixtures/toy_chunk_f2_reference.csv)

Gồm 4 queries ($N=4$):

### Query 1 (`toy_q1`):
- $T = \{c_1, c_2\} \implies |T| = 2$
- $P = \{c_1, c_3\} \implies |P| = 2$
- $TP = |\{c_1\}| = 1$, $FP = |\{c_3\}| = 1$, $FN = |\{c_2\}| = 1$
- $\text{Precision} = \frac{1}{2} = 0.5$
- $\text{Recall} = \frac{1}{2} = 0.5$
- $F_1 = \frac{2 \cdot 1}{2 + 2} = \frac{2}{4} = 0.5$
- $F_2 = \frac{5 \cdot 1}{4 \cdot 2 + 2} = \frac{5}{10} = 0.5$

### Query 2 (`toy_q2`):
- $T = \{c_4\} \implies |T| = 1$
- $P = \{c_4, c_5, c_6\} \implies |P| = 3$
- $TP = |\{c_4\}| = 1$, $FP = |\{c_5, c_6\}| = 2$, $FN = 0$
- $\text{Precision} = \frac{1}{3} \approx 0.333333$
- $\text{Recall} = \frac{1}{1} = 1.0$
- $F_1 = \frac{2 \cdot 1}{1 + 3} = \frac{2}{4} = 0.5$
- $F_2 = \frac{5 \cdot 1}{4 \cdot 1 + 3} = \frac{5}{7} \approx 0.714286$

### Query 3 (`toy_q3`):
- $T = \{c_7, c_8\} \implies |T| = 2$
- $P = \{c_9\} \implies |P| = 1$
- $TP = 0$, $FP = 1$, $FN = 2$
- $\text{Precision} = 0.0, \quad \text{Recall} = 0.0, \quad F_1 = 0.0, \quad F_2 = 0.0$

### Query 4 (`toy_q4`):
- $T = \{c_{10}\} \implies |T| = 1$
- $P = \emptyset \implies |P| = 0$
- $TP = 0$, $FP = 0$, $FN = 1$
- $\text{Precision} = 0.0, \quad \text{Recall} = 0.0, \quad F_1 = 0.0, \quad F_2 = 0.0$

### Tổng hợp Macro Toy:
- $\text{Macro Precision} = \frac{0.5 + 1/3 + 0 + 0}{4} = \frac{5/6}{4} = \frac{5}{24} \approx 0.208333$
- $\text{Macro Recall} = \frac{0.5 + 1.0 + 0 + 0}{4} = \frac{1.5}{4} = \frac{3}{8} = 0.375000$
- $\text{Macro } F_1 = \frac{0.5 + 0.5 + 0 + 0}{4} = \frac{1.0}{4} = 0.250000$
- $\text{Macro } F_2 = \frac{0.5 + 5/7 + 0 + 0}{4} = \frac{17/14}{4} = \frac{17}{56} \approx 0.303571$

---

## 3. Tính tay trên bộ dữ liệu thực tế `data/p1_p2_handoff_qwen3`

Tệp phụ: [`data/p1_p2_handoff_qwen3/manual_verification_f2.csv`](../data/p1_p2_handoff_qwen3/manual_verification_f2.csv)

Bộ dữ liệu gồm $N=7$ queries, mỗi query có đúng 1 ground truth chunk ($|T|=1$). Vị trí rank xuất hiện của ground truth chunk trong danh sách candidates của Người 1:
- `prototype_query_8b6b...`: rank 1 (trúng)
- `prototype_query_1a44...`: rank 2
- `prototype_query_2823...`: rank 2
- `prototype_query_a2df...`: rank 1 (trúng)
- `prototype_query_153f...`: rank 10
- `prototype_query_330b...`: rank 10
- `prototype_query_d724...`: rank 3

### Kịch bản A: Lấy Top-1 candidate ($k=1$)
Mỗi query lấy đúng 1 candidate có score cao nhất ($|P|=1$).
- Có 2 queries trúng ở rank 1: `prototype_query_8b6b...` và `prototype_query_a2df...`.
  - Với 2 queries này: $TP=1, |T|=1, |P|=1 \implies \text{Precision}=1.0, \text{Recall}=1.0, F_1=1.0, F_2=1.0$.
- Có 5 queries còn lại trượt: $TP=0 \implies \text{Precision}=0.0, \text{Recall}=0.0, F_1=0.0, F_2=0.0$.
- **Macro Top-1**:
  $$\text{Macro Precision} = \frac{1 + 0 + 0 + 1 + 0 + 0 + 0}{7} = \frac{2}{7} \approx 0.285714$$
  $$\text{Macro Recall} = \frac{2}{7} \approx 0.285714$$
  $$\text{Macro } F_1 = \frac{2}{7} \approx 0.285714$$
  $$\text{Macro } F_2 = \frac{2}{7} \approx 0.285714$$

### Kịch bản B: Lấy Top-2 candidates ($k=2$)
Mỗi query lấy 2 candidates có score cao nhất ($|P|=2$).
- Có 4 queries trúng (rank 1 hoặc rank 2):
  - `prototype_query_8b6b...` (rank 1)
  - `prototype_query_1a44...` (rank 2)
  - `prototype_query_2823...` (rank 2)
  - `prototype_query_a2df...` (rank 1)
  Với mỗi query này:
  $TP=1, |T|=1, |P|=2$
  - $\text{Precision} = \frac{1}{2} = 0.5$
  - $\text{Recall} = \frac{1}{1} = 1.0$
  - $F_1 = \frac{2 \cdot 1}{1 + 2} = \frac{2}{3} \approx 0.666667$
  - $F_2 = \frac{5 \cdot 1}{4 \cdot 1 + 2} = \frac{5}{6} \approx 0.833333$
- Có 3 queries còn lại trượt (ground truth ở rank 3, 10, 10):
  - $\text{Precision} = 0.0, \quad \text{Recall} = 0.0, \quad F_1 = 0.0, \quad F_2 = 0.0$
- **Macro Top-2**:
  $$\text{Macro Precision} = \frac{4 \times 0.5 + 3 \times 0}{7} = \frac{2}{7} \approx 0.285714$$
  $$\text{Macro Recall} = \frac{4 \times 1.0 + 3 \times 0}{7} = \frac{4}{7} \approx 0.571429$$
  $$\text{Macro } F_1 = \frac{4 \times \frac{2}{3} + 3 \times 0}{7} = \frac{8}{21} \approx 0.380952$$
  $$\text{Macro } F_2 = \frac{4 \times \frac{5}{6} + 3 \times 0}{7} = \frac{20/6}{7} = \frac{10}{21} \approx 0.476190$$

---

## 4. Hướng dẫn chạy và đối chiếu

### Chạy CLI kiểm tra:
```bash
# Đánh giá Top-1 trên data/p1_p2_handoff_qwen3
python scripts/evaluate_f2.py --handoff-dir data/p1_p2_handoff_qwen3 --top-k 1

# Đánh giá Top-2 trên data/p1_p2_handoff_qwen3
python scripts/evaluate_f2.py --handoff-dir data/p1_p2_handoff_qwen3 --top-k 2

# Xuất ra CSV để mở trong Excel
python scripts/evaluate_f2.py --handoff-dir data/p1_p2_handoff_qwen3 --top-k 2 --output-csv outputs/p2_02_eval_top2.csv
```

### Chạy automated unit tests:
```bash
python -m pytest tests/test_evaluate_f2.py -v
```
Tất cả các assertions so sánh trực tiếp với các phân số chính xác $\frac{5}{24}, \frac{3}{8}, \frac{1}{4}, \frac{17}{56}$ (cho toy) và $\frac{2}{7}, \frac{4}{7}, \frac{8}{21}, \frac{10}{21}$ (cho handoff real data) bằng `pytest.approx(..., abs=1e-6)`.
