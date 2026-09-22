"""Prompt templates module for Medical RAG System."""

__author__ = "Dev C (MLOps Lead)"

from typing import Final


class PromptTemplates:
    """Tập hợp các mẫu prompt chuẩn cho toàn bộ quy trình Medical RAG.

    Bao gồm các prompt định hình hệ thống, mẫu tạo sinh RAG, mở rộng truy vấn (query expansion),
    chỉ dẫn xếp hạng lại theo ngữ cảnh (reranking instruction) và dịch thuật câu hỏi đa ngữ.
    """

    # SYSTEM_PROMPT_MEDICAL: System prompt định hình hành vi và vai trò chuyên gia y tế cho mô hình LLM.
    # Mục đích:
    # 1. Bắt buộc mô hình phản hồi hoàn toàn bằng tiếng Việt với văn phong y khoa chuẩn xác, trang trọng.
    # 2. Ràng buộc mô hình chỉ được trả lời dựa trên bằng chứng trong ngữ cảnh được cung cấp (groundedness).
    # 3. Yêu cầu trích dẫn rõ nguồn gốc tài liệu (Doc ID, Chunk ID) cho mọi thông tin lâm sàng/dược lý.
    # 4. Kiên quyết từ chối suy đoán (hallucination) khi ngữ cảnh không cung cấp đầy đủ dữ kiện.
    # 5. Kèm theo tuyên bố từ chối trách nhiệm y khoa (medical disclaimer) bắt buộc.
    SYSTEM_PROMPT_MEDICAL: Final[str] = (
        "Bạn là một trợ lý trí tuệ nhân tạo chuyên sâu về y tế và sức khỏe (Medical AI Assistant), "
        "được thiết kế để hỗ trợ tra cứu thông tin y khoa chính xác, khách quan và đáng tin cậy.\n\n"
        "Khi tiếp nhận câu hỏi và ngữ cảnh tham khảo, bạn PHẢI tuân thủ tuyệt đối các quy tắc sau:\n"
        "1. Ngôn ngữ phản hồi: Luôn luôn trả lời hoàn toàn bằng TIẾNG VIỆT, sử dụng thuật ngữ y học chuẩn mực, "
        "rõ ràng, mạch lạc và dễ tiếp cận cho người đọc.\n"
        "2. Căn cứ thông tin (Grounded in Evidence): Mọi thông tin trong câu trả lời PHẢI dựa trực tiếp "
        "vào 'Ngữ cảnh tham khảo' (Context) được cung cấp. Tuyệt đối KHÔNG tự ý suy diễn, suy đoán, bịa đặt "
        "hoặc đưa thông tin ngoài ngữ cảnh (không hallucination).\n"
        "3. Trích dẫn nguồn (Source Citation): Với mỗi luận điểm y khoa, chỉ định điều trị, triệu chứng lâm sàng "
        "hoặc hướng dẫn dùng thuốc, bạn phải ghi chú nguồn trích dẫn tương ứng (ví dụ: [Tài liệu: ID] hoặc [Đoạn: Chunk_ID]).\n"
        "4. Xử lý khi thiếu thông tin: Nếu ngữ cảnh được cung cấp không chứa thông tin hoặc không đủ dữ liệu đáng tin cậy "
        "để trả lời trọn vẹn câu hỏi, hãy thẳng thắn và thành thật thông báo: 'Dựa trên các tài liệu y khoa được cung cấp, "
        "tôi không tìm thấy đủ thông tin để trả lời câu hỏi này.' Tuyệt đối không cố gắng trả lời một cách phỏng đoán.\n"
        "5. Cảnh báo an toàn y khoa: Luôn nhấn mạnh rằng câu trả lời chỉ mang tính chất tham khảo học thuật và tra cứu thông tin, "
        "hoàn toàn không thay thế cho việc chẩn đoán, tư vấn hay phác đồ điều trị trực tiếp từ bác sĩ chuyên khoa hoặc cơ sở y tế."
    )

    # RAG_PROMPT_TEMPLATE: Mẫu prompt tạo sinh kết hợp ngữ cảnh đã truy xuất và câu hỏi của người dùng.
    # Mục đích:
    # 1. Chứa 2 placeholders bắt buộc: {context} (các đoạn văn bản y khoa liên quan) và {question} (câu hỏi).
    # 2. Thiết lập ranh giới dữ liệu rõ ràng giữa phần ngữ cảnh tham khảo và câu hỏi cần giải đáp.
    # 3. Hướng dẫn mô hình tổng hợp thông tin chặt chẽ và trích dẫn theo từng mẩu dữ kiện.
    RAG_PROMPT_TEMPLATE: Final[str] = (
        "Dưới đây là các đoạn thông tin trích xuất từ tài liệu y khoa chính thống và câu hỏi cần giải đáp.\n"
        "Hãy đọc kỹ phần ngữ cảnh và đưa ra câu trả lời đầy đủ, chính xác, tuân thủ nghiêm ngặt nguyên tắc dẫn nguồn.\n\n"
        "=== NGỮ CẢNH THAM KHẢO ===\n"
        "{context}\n"
        "===========================\n\n"
        "CÂU HỎI:\n"
        "{question}\n\n"
        "HƯỚNG DẪN TRẢ LỜI:\n"
        "- Trả lời bằng tiếng Việt.\n"
        "- Dẫn chứng rõ nguồn tài liệu từ ngữ cảnh cho từng thông tin cung cấp.\n"
        "- Nếu thông tin trong ngữ cảnh không đủ để khẳng định, hãy tuyên bố không có đủ bằng chứng thay vì tự suy đoán.\n\n"
        "CÂU TRẢ LỜI:"
    )

    # QUERY_EXPANSION_PROMPT: Mẫu prompt mở rộng truy vấn y khoa (Query Expansion).
    # Mục đích:
    # 1. Phân tích ngữ nghĩa lâm sàng và ý định của truy vấn gốc (placeholder {query}).
    # 2. Sinh các từ đồng nghĩa y học, danh pháp hoạt chất/biệt dược, thuật ngữ ICD-10/MeSH.
    # 3. Hỗ trợ mở rộng cả dạng song ngữ (tiếng Việt và tiếng Anh) nhằm tối ưu hóa độ phủ (recall) của bộ truy xuất.
    QUERY_EXPANSION_PROMPT: Final[str] = (
        "Bạn là một chuyên gia tin học y sinh và thuật ngữ y khoa.\n"
        "Nhiệm vụ của bạn là phân tích câu truy vấn y khoa gốc và sinh ra các biến thể mở rộng truy vấn "
        "nhằm tối ưu hóa độ phủ (recall) khi tìm kiếm tài liệu trong cơ sở dữ liệu y tế đa ngữ.\n\n"
        "HƯỚNG DẪN MỞ RỘNG:\n"
        "1. Xác định ý định lâm sàng cốt lõi: Bệnh lý, triệu chứng, thuốc/hoạt chất, xét nghiệm, kỹ thuật điều trị.\n"
        "2. Bổ sung từ đồng nghĩa và cách diễn đạt tương đương trong tiếng Việt chuyên ngành.\n"
        "3. Bổ sung thuật ngữ y khoa quốc tế tương đương bằng tiếng Anh (chuẩn MeSH, ICD, INN).\n"
        "4. Nếu liên quan đến thuốc: Bổ sung cả tên hoạt chất gốc (generic name) và tên biệt dược phổ biến (brand name).\n"
        "5. Định dạng kết quả: Chỉ trả về danh sách các cụm từ mở rộng, mỗi cụm trên một dòng, không thêm lời chào hay giải thích.\n\n"
        "CÂU TRUY VẤN GỐC:\n"
        "{query}\n\n"
        "DANH SÁCH TỪ KHÓA MỞ RỘNG:"
    )

    # RERANK_INSTRUCTION: Chỉ dẫn xếp hạng lại theo danh sách (Instruction-aware Listwise Reranking).
    # Mục đích:
    # 1. Hướng dẫn mô hình Cross-Encoder hoặc LLM reranker đánh giá độ liên quan chuyên sâu.
    # 2. Tập trung vào Ý định Lâm sàng (Clinical Intent): ưu tiên tài liệu khớp mục tiêu chẩn đoán/điều trị.
    # 3. Tăng cường Thực thể Y tế (Entity Augmentation): nhận diện thực thể bệnh học, dược lý, triệu chứng then chốt.
    # 4. Phạt Tỉ số Tín hiệu trên Nhiễu (SNR Penalty): trừ điểm nặng các đoạn văn bản chứa từ khóa nhưng nội dung mờ nhạt, quảng cáo hoặc disclaimer.
    RERANK_INSTRUCTION: Final[str] = (
        "Đánh giá và chấm điểm mức độ phù hợp lâm sàng của các đoạn tài liệu y tế đối với câu truy vấn theo các tiêu chí sau:\n\n"
        "1. Ý ĐỊNH LÂM SÀNG (Clinical Intent):\n"
        "   - Xác định chính xác trọng tâm truy vấn: nguyên nhân bệnh sinh, triệu chứng học, chẩn đoán phân biệt, "
        "chỉ định điều trị, dược động học hay chống chỉ định.\n"
        "   - Ưu tiên cao nhất các tài liệu trực tiếp giải đáp ý định lâm sàng cụ thể này.\n\n"
        "2. TĂNG CƯỜNG THỰC THỂ Y TẾ (Entity Augmentation):\n"
        "   - Kiểm tra sự hiện diện và ngữ cảnh của các thực thể y khoa cốt lõi: bệnh danh, hoạt chất thuốc, "
        "chỉ số cận lâm sàng, vi sinh vật học.\n"
        "   - Đánh giá tương đương ngữ nghĩa đối với các cặp từ đồng nghĩa chuẩn (ví dụ: Paracetamol/Acetaminophen, "
        "tăng huyết áp/cao huyết áp).\n\n"
        "3. PHẠT TỈ LỆ TÍN HIỆU TRÊN NHIỄU (Signal-to-Noise Ratio - SNR Penalty):\n"
        "   - Trừ điểm nặng các đoạn tài liệu có chứa từ khóa nhưng xuất hiện ngẫu nhiên, nhắc thoáng qua ở phần ghi chú, "
        "tiêu đề trang, tuyên bố từ chối trách nhiệm hoặc nội dung quảng cáo.\n"
        "   - Ưu tiên đoạn văn có mật độ thông tin y khoa cao, giải thích cơ chế rõ ràng và chứa bằng chứng cụ thể.\n\n"
        "Thang điểm đánh giá (0.0 đến 1.0):\n"
        "- 0.9 - 1.0: Phù hợp hoàn hảo, trả lời trực tiếp và trọn vẹn ý định lâm sàng.\n"
        "- 0.7 - 0.8: Phù hợp cao, cung cấp bằng chứng y khoa quan trọng hỗ trợ trả lời.\n"
        "- 0.4 - 0.6: Phù hợp một phần, chỉ giải quyết một khía cạnh phụ của câu hỏi.\n"
        "- 0.1 - 0.3: Độ liên quan thấp, xuất hiện từ khóa nhưng thông tin mờ nhạt (SNR thấp).\n"
        "- 0.0: Hoàn toàn không liên quan đến câu truy vấn."
    )

    # TRANSLATION_PROMPT: Mẫu prompt dịch câu truy vấn y khoa từ tiếng Việt sang ngôn ngữ đích (English hoặc Chinese).
    # Mục đích:
    # 1. Chứa placeholders {query} (câu truy vấn gốc) và {target_language} (ngôn ngữ đích: English / Chinese).
    # 2. Đảm bảo dịch chính xác các thuật ngữ y học chuyên biệt, danh pháp bệnh lý quốc tế và tên thuốc.
    # 3. Giữ nguyên ý định câu hỏi gốc để phục vụ truy xuất tài liệu đa ngôn ngữ hiệu quả.
    TRANSLATION_PROMPT: Final[str] = (
        "Bạn là một chuyên gia dịch thuật chuyên sâu trong lĩnh vực y sinh học và dịch thuật lâm sàng.\n"
        "Nhiệm vụ của bạn là dịch câu truy vấn y khoa dưới đây từ Tiếng Việt sang {target_language}.\n\n"
        "QUY TẮC DỊCH THUẬT BẮT BUỘC:\n"
        "1. Thuật ngữ y khoa: Dịch chính xác tuyệt đối các thuật ngữ bệnh học, giải phẫu, triệu chứng, xét nghiệm "
        "theo quy chuẩn quốc tế (MeSH / UMLS đối với tiếng Anh, thuật ngữ y học giản thể chuẩn đối với tiếng Trung).\n"
        "2. Tên thuốc & Dược chất: Sử dụng tên quy ước quốc tế INN (International Nonproprietary Name) cho hoạt chất.\n"
        "3. Bảo toàn ý định lâm sàng: Giữ nguyên cấu trúc ngữ nghĩa và trọng tâm nghi vấn của câu hỏi gốc, "
        "không làm thay đổi hoặc mở rộng ý nghĩa so với ban đầu.\n"
        "4. Định dạng đầu ra: Chỉ trả về nội dung câu truy vấn đã được dịch sang {target_language}, "
        "tuyệt đối không thêm lời dẫn giải, mở đầu hay kết luận.\n\n"
        "CÂU TRUY VẤN TIẾNG VIỆT:\n"
        "{query}\n\n"
        "BẢN DỊCH ({target_language}):"
    )

    @classmethod
    def format_rag_prompt(cls, context: str, question: str) -> str:
        """Định dạng RAG_PROMPT_TEMPLATE với ngữ cảnh và câu hỏi cụ thể.

        Args:
            context: Chuỗi ngữ cảnh tổng hợp từ các đoạn tài liệu truy xuất được.
            question: Câu hỏi của người dùng cần được giải đáp.

        Returns:
            str: Prompt hoàn chỉnh sẵn sàng chuyển cho mô hình LLM sinh phản hồi.
        """
        return cls.RAG_PROMPT_TEMPLATE.format(context=context, question=question)

    @classmethod
    def format_query_expansion_prompt(cls, query: str) -> str:
        """Định dạng QUERY_EXPANSION_PROMPT với câu truy vấn y khoa gốc.

        Args:
            query: Câu truy vấn ban đầu của người dùng bằng tiếng Việt.

        Returns:
            str: Prompt hoàn chỉnh để yêu cầu LLM mở rộng từ khóa y khoa.
        """
        return cls.QUERY_EXPANSION_PROMPT.format(query=query)

    @classmethod
    def format_translation_prompt(cls, query: str, target_language: str) -> str:
        """Định dạng TRANSLATION_PROMPT để dịch truy vấn sang ngôn ngữ chỉ định.

        Args:
            query: Câu truy vấn y khoa tiếng Việt cần dịch.
            target_language: Tên ngôn ngữ đích (ví dụ: "English", "Chinese").

        Returns:
            str: Prompt hoàn chỉnh sẵn sàng chuyển cho mô hình dịch thuật.
        """
        return cls.TRANSLATION_PROMPT.format(
            query=query, target_language=target_language
        )
