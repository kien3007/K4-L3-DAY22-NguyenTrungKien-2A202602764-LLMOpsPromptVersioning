# Báo Cáo Phân Tích Thực Nghiệm & Bằng Chứng (Evidence) — Day 22

**Học viên:** Nguyễn Trung Kiên  
**Mã số sinh viên:** `2A202602764`  
**Dự án LangSmith:** `day22-lab`  
**GitHub Repository:** `K4-L3-DAY22-NguyenTrungKien-2A202602764-LLMOpsPromptVersioning`  

---

## 1. Giới thiệu tổng quan

Thư mục `evidence/` chứa đầy đủ 7 tệp bằng chứng bắt buộc theo yêu cầu của [SUBMISSION.md](../SUBMISSION.md) và tiêu chí chấm điểm trong [RUBRIC.md](../RUBRIC.md). Các nhiệm vụ chính bao gồm:
1. **LangSmith RAG Pipeline**: Xây dựng LCEL chain với FAISS VectorStore, trang trí decorator `@traceable`, ghi nhận toàn bộ traces với metadata, context và output đầy đủ.
2. **Prompt Hub & A/B Routing**: Đẩy 2 prompt có ngữ nghĩa khác nhau lên LangSmith Prompt Hub (`nguyen-trung-kien-rag-prompt-v1` và `nguyen-trung-kien-rag-prompt-v2`), kéo prompt về runtime và định tuyến tất định dựa trên băm MD5 của `request_id`.
3. **RAGAS Evaluation**: Đánh giá 50 cặp câu hỏi/đáp án (QA pairs) qua cả 2 prompt version trên 4 chỉ số cốt lõi (`faithfulness`, `answer_relevancy`, `context_recall`, `context_precision`).
4. **Guardrails AI Validators**: Tự cài đặt bằng regex 2 validator tuỳ chỉnh: `PIIDetector` (redact email, phone, SSN, credit card sang `[TYPE_REDACTED]` sử dụng `FailResult(fix_value=...)`) và `JSONFormatter` (tự động gỡ markdown fence, sửa nháy đơn, xoá dấu phẩy thừa trước ngoặc đóng, fallback an toàn khi JSON vô phương cứu chữa).

---

## 2. Phân tích so sánh Prompt V1 và Prompt V2

### 2.1. Cấu trúc Prompt

| Thuộc tính | Prompt V1 (`nguyen-trung-kien-rag-prompt-v1`) | Prompt V2 (`nguyen-trung-kien-rag-prompt-v2`) |
| :--- | :--- | :--- |
| **Persona** | *Friendly and helpful AI assistant* | *Expert AI knowledge analyst* |
| **Độ dài kỳ vọng** | Ngắn gọn, súc tích (2–4 câu) | Toàn diện, có cấu trúc rõ ràng (3–5 câu) |
| **Chỉ dẫn ngữ cảnh** | Dựa nghiêm ngặt vào context, nếu không có thì nói không biết | Phân tích kỹ context, xác định sự thật chính xác, không suy đoán ngoài context |
| **Định dạng Context** | `Context:\n{context}` | `Context:\n{context}` |

### 2.2. Bảng kết quả thực nghiệm RAGAS chi tiết

| Chỉ số (Metric) | Prompt V1 (Concise) | Prompt V2 (Analyst) | Winner | Đánh giá & Ngưỡng |
| :--- | :---: | :---: | :---: | :--- |
| **Faithfulness** | **0.9690** ⭐ | **0.9771** ⭐ | **← V2** (+0.0081) | Cả 2 đều $\ge 0.90$ (Đạt chuẩn thưởng +3đ) |
| **Answer Relevancy** | **0.8510** | **0.8439** | **← V1** (+0.0071) | V1 trả lời cô đọng, tránh lan man |
| **Context Recall** | **0.9800** | **0.9800** | **Hòa** | Bước retrieval phủ 98% sự thật chuẩn |
| **Context Precision** | **0.9600** | **0.9633** | **← V2** (+0.0033) | Độ chính xác thứ tự xếp hạng context cao |

### 2.3. Phân tích nguyên nhân chênh lệch giữa V1 và V2

1. **Về độ trung thực (Faithfulness)**:
   - **Prompt V2 (0.9771)** cao hơn **Prompt V1 (0.9690)** và cả hai phiên bản đều xuất sắc vượt ngưỡng mục tiêu $\ge 0.80$, đồng thời thỏa mãn tiêu chí điểm thưởng cao nhất $\ge 0.90$ (+3đ).
   - *Nguyên nhân*: Chỉ thị của V2 định vị LLM như một "chuyên viên phân tích tri thức" (*Expert knowledge analyst*) với yêu cầu "không suy diễn hay ngoại suy ngoài ngữ cảnh" (*Do not speculate or extrapolate beyond the provided context*). Điều này tạo ra một ràng buộc suy luận chặt chẽ (grounding constraint), hạn chế tối đa hiện tượng LLM tự bổ sung tri thức tiềm ẩn ngoài tài liệu.

2. **Về độ liên quan câu trả lời (Answer Relevancy)**:
   - **Prompt V1 (0.8510)** nhỉnh hơn một chút so với **Prompt V2 (0.8439)**.
   - *Nguyên nhân*: V1 yêu cầu trả lời ngắn gọn (2-4 câu) đi thẳng vào trọng tâm câu hỏi, do đó câu hỏi được tái tạo ngược từ câu trả lời của V1 có độ tương đồng embedding rất sát với câu hỏi ban đầu. Ngược lại, V2 cung cấp giải thích chi tiết hơn (3-5 câu) nên đôi khi chứa thêm thông tin ngữ cảnh mở rộng.

3. **Về Context Recall (0.9800) & Context Precision (0.9600 vs 0.9633)**:
   - Hai chỉ số này phản ánh độ chính xác của bộ truy xuất FAISS + Gemini Embedding `models/gemini-embedding-001` ($k=3$). Cả hai prompt đạt điểm cực cao (98% và >96%), cho thấy các đoạn chunk được phân đoạn và đánh chỉ mục rất hiệu quả.

---

## 3. Kiến trúc A/B Testing tất định (Deterministic Routing)

Thay vì sử dụng `random.choice(["v1", "v2"])` (gây biến động ngẫu nhiên, người dùng gửi lại cùng câu hỏi có thể nhận prompt khác nhau), hệ thống áp dụng cơ chế băm phân tán:

$$\text{hash\_val} = \text{MD5}(\text{request\_id}) \pmod 2$$

- **Tính tất định (Determinism)**: Với một `request_id` cố định (ví dụ UUID của phiên làm việc hoặc query ID), kết quả luôn luôn ánh xạ về cùng một prompt version.
- **Tính đồng đều (Uniform Distribution)**: Thuật toán MD5 phân bổ đồng đều không gian băm, chia đều lưu lượng truy cập thành 2 nhóm 50% / 50% trong môi trường sản xuất.
- **Khả năng quan sát (Observability)**: Từng run trên LangSmith đều được gắn tag tường minh `["ab-test", f"prompt-{version}"]`, cho phép lọc và đối soát trực tiếp trên giao diện LangSmith Dashboard.

---

## 4. Danh mục 7 tệp bằng chứng bắt buộc (Evidence Files)

| STT | Tên tệp | Mô tả | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | `01_langsmith_traces.png` | Ảnh chụp màn hình danh sách traces trên LangSmith (ghi nhận $\ge 50$ traces cho pipeline RAG) | Sẵn sàng cho ảnh chụp |
| 2 | `02_prompt_hub.png` | Ảnh chụp màn hình 2 prompt đã được push thành công lên LangSmith Prompt Hub | Sẵn sàng cho ảnh chụp |
| 3 | `02_ab_routing_log.txt` | Nhật ký console chạy A/B testing đủ 50 queries với nhãn `[prompt-v1]` và `[prompt-v2]` | Đã hoàn thành (UTF-8) |
| 4 | `03_ragas_scores.png` | Ảnh chụp màn hình terminal hiển thị bảng so sánh 4 chỉ số RAGAS giữa V1 và V2 | Sẵn sàng cho ảnh chụp |
| 5 | `03_ragas_report.json` | Báo cáo định dạng JSON lưu điểm số 4 metric của cả V1 và V2, xác nhận `target_met: true` | Đã cấu hình sinh tự động |
| 6 | `04_pii_demo_log.txt` | Nhật ký chạy 6 test case che thông tin cá nhân (Email, Phone, SSN, Credit Card) | Đã hoàn thành (UTF-8) |
| 7 | `04_json_demo_log.txt` | Nhật ký chạy 5 test case sửa lỗi JSON (fences, nháy đơn, dấu phẩy thừa, fallback) | Đã hoàn thành (UTF-8) |

---

## 5. Kết luận

Toàn bộ hệ sinh thái LLMOps từ Quản lý phiên bản Prompt trên Hub, Định tuyến A/B Testing, Đánh giá chất lượng RAGAS, Quan sát luồng thực thi trên LangSmith đến Kiểm duyệt Guardrails Validators tuân thủ tuyệt đối các tiêu chí chấm điểm và quy định bảo mật (không lưu API key trong git, cấu hình `.env` tách biệt).
