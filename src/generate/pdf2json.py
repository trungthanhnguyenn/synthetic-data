import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.utils.utils import get_filename_without_ext
import getpass
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

if "GOOGLE_API_KEY" not in os.environ:
    os.getenv("GOOGLE_API_KEY")
    
SYSTEM_PROMPT_LAW_EXTRACTION = ChatPromptTemplate.from_messages([
    ("system", r"""
Bạn là một chuyên gia pháp luật có nhiệm vụ **trích xuất thông tin có cấu trúc** từ văn bản pháp luật đã được số hóa (OCR hoặc định dạng văn bản thường).

Yêu cầu:
- Dựa vào phần `context` bên dưới, hãy điền thông tin vào **hai bảng JSON** với đúng **tên trường và định dạng** như mô tả.
- Nếu **không đủ thông tin** để điền các bảng, hãy trả về **chuỗi duy nhất**: `PDF không chứa đủ thông tin để điền vào bảng.`
- Tuyệt đối **không trả lời, giải thích hoặc tạo thêm thông tin ngoài context.**
- Chỉ trả về đúng hai bảng JSON hoặc chuỗi đặc biệt trên.

------ 🗂️ Bảng 1: Thông tin văn bản pháp luật ------
```json
{{
  "so_hieu": "...",                  // Số hiệu văn bản (VD: 01/2023/TT-BGDĐT)
  "loai_vb": "...",                  // Loại văn bản (VD: Thông tư, Nghị định)
  "noi_ban_hanh": "...",            // Cơ quan ban hành (VD: Bộ Tài chính)
  "nguoi_ky": "...",                // Người ký ban hành
  "ngay_ban_hanh": "...",           // Ngày ban hành (YYYY-MM-DD)
  "ngay_hieu_luc": "...",           // Ngày có hiệu lực
  "ngay_cong_bao": "...",           // Ngày công bố
  "so_cong_bao": "...",             // Số Công báo
  "tinh_trang": "...",              // Trạng thái hiệu lực (VD: Còn hiệu lực)
  "tieu_de": "...",                 // Tên văn bản
  "noi_dung": "...",                // Nội dung chính của văn bản
  "linh_vuc": "..."                 // Lĩnh vực (VD: Giáo dục, Y tế)
}}
------ Bảng 2: Văn bản liên quan ------
{{
  "tieu_de": "...",                         // Trùng với tiêu đề văn bản hiện tại
  "vb_duoc_hd": ["..."],                    // Văn bản được hướng dẫn bởi
  "vb_hd": ["..."],                         // Văn bản hướng dẫn cho
  "vb_bi_sua_doi_bo_sung": ["..."],         // Văn bản bị sửa đổi, bổ sung bởi
  "vb_sua_doi_bo_sung": ["..."],            // Văn bản sửa đổi, bổ sung cho
  "vb_duoc_hop_nhat": ["..."],              // Văn bản được hợp nhất vào
  "vb_hop_nhat": ["..."],                   // Văn bản hợp nhất các văn bản khác
  "vb_bi_dinh_chinh": ["..."],              // Văn bản bị đính chính bởi
  "vb_dinh_chinh": ["..."],                 // Văn bản đính chính cho
  "vb_bi_thay_the": ["..."],                // Văn bản bị thay thế bởi
  "vb_thay_the": ["..."],                   // Văn bản thay thế cho
  "vb_duoc_dan_chieu": ["..."],             // Văn bản được dẫn chiếu bởi
  "vb_duoc_can_cu": ["..."],                // Văn bản được căn cứ bởi
  "vb_lien_quan_cung_noi_dung": ["..."]     // Các văn bản liên quan nội dung
}}
"""),
    ("human", "Context:\n{context}\n\nQuestion:\nHãy trích xuất thông tin theo yêu cầu.")
])

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=32000,
    timeout=30
)

def process_txt_with_gemini(text_data: str):
    chain = SYSTEM_PROMPT_LAW_EXTRACTION | llm
    response = chain.invoke({"context": text_data})
    return response.content

def generate_json(in_txt: str, output_dir: str):
    with open(in_txt, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    formatted_data = process_txt_with_gemini(raw_text)
    base_name = get_filename_without_ext(in_txt)

    # Kiểm tra nội dung trả về có phải chuỗi cảnh báo không
    if formatted_data.strip() == "PDF không chứa đủ thông tin để điền vào bảng.":
        target_dir = os.path.join(output_dir, "fail")
    else:
        target_dir = output_dir

    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)

    out_json = os.path.join(target_dir, f"{base_name}.json")

    with open(out_json, 'w', encoding='utf-8') as f:
        f.write(formatted_data)
        
    print(f"Đã lưu JSON vào: {out_json}")
    
# if __name__=='__main__':
#     generate_json('/home/truongnn/trung/project/synthetic_data/data/input/txt/Báo cáo 191_BC-BTTTT năm 2024 đánh giá tình hình hoạt động của cơ sở truyền thanh - truyền hình cấp huyện do Bộ Thông tin và Truyền thông ban hành.txt', '/home/truongnn/trung/project/synthetic_data/data/output')