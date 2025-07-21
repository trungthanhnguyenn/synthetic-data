import os
from tqdm import tqdm
import csv
from huggingface_hub import HfApi, HfFolder, upload_file
from datasets import load_dataset, Dataset, DatasetDict
import sys

system_prompt = r"""
Bạn là một chuyên gia pháp luật có nhiệm vụ **trích xuất thông tin có cấu trúc** từ văn bản pháp luật đã được số hóa (OCR hoặc định dạng văn bản thường).

Yêu cầu:
- Dựa vào phần context bên dưới, hãy điền thông tin vào **hai bảng JSON** với đúng **tên trường và định dạng** như mô tả.
- Nếu **không đủ thông tin** để điền các bảng, hãy trả về **chuỗi duy nhất**: PDF không chứa đủ thông tin để điền vào bảng.
- Tuyệt đối **không trả lời, giải thích hoặc tạo thêm thông tin ngoài context.**
- Chỉ trả về đúng hai bảng JSON hoặc chuỗi đặc biệt trên.
- Không tóm tắt phần "noi_dung" mà phải trả về **toàn bộ** nội dung của văn bản.

------ 🗂️ Bảng 1: Thông tin văn bản pháp luật ------
json
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
  "noi_dung": "...",                // Toàn bộ nội dung của văn bản không được tóm tắt
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
"""

human_prompt = "Hãy trích xuất thông tin theo yêu cầu."

def get_filename_without_ext(file_path: str) -> str:
    """
    Get the filename without extension from a file path.
    Example: /path/to/file.pdf -> file
    """
    base = os.path.basename(file_path)
    name, _ = os.path.splitext(base)
    return name

def get_all_env_values():
    keys = [
        "GPT_API_KEY",
        "GPT_MODEL_NAME",
        "GPT_BASE_URL",
        "OPENROUTER_KEY",
        "OPEN_ROUTER_NAME",
        "OPENROUTER_BASE_URL",
        "DEEPSEEK_KEY",
        "DEEPSEEK_MODEL_NAME",
        "DEEPSEEK_BASE_URL",
        "GROQ_KEY",
        "GROQ_MODEL_NAME",
        "GROQ_BASE_URL",
        "NVIDIA_KEY",
        "NVIDIA_NAME",
        "NVIDIA_BASE_URL",
        "GITHUB_KEY",
        "GITHUB_VALUE",
        "GITHUB_MODEL_NAME",
        "GEMINI_KEY",
        "GEMINI_MODEL_NAME"
    ]
    return {key: os.getenv(key) for key in keys}

def make_csv(output_csv, txt_path):
  with open(output_csv, mode="w", newline='', encoding="utf-8") as csvfile:
      writer = csv.DictWriter(
          csvfile,
          fieldnames=["title", "system", "human", "context"],
          quoting=csv.QUOTE_ALL
      )
      writer.writeheader()

      for filename in os.listdir(txt_path):
          if filename.endswith(".txt"):
              file_path = os.path.join(txt_path, filename)
              with open(file_path, "r", encoding="utf-8") as f:
                  raw_context = f.read().strip().replace('"', '""')  # escape dấu "
                  title = os.path.splitext(filename)[0]

                  writer.writerow({
                      "title": title,
                      "system": system_prompt.strip(),
                      "human": human_prompt,
                      "context": raw_context
                  })

  print(f"✅ File CSV đã được tạo thành công tại: {output_csv}")
  
def upload(csv_path, repo_id):
  dataset = Dataset.from_csv(csv_path)
  data = DatasetDict({'train': dataset})
  data.push_to_hub(repo_id=repo_id, max_shard_size='150MB')