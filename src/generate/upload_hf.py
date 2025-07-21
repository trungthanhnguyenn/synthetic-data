import os
from tqdm import tqdm
import csv
from huggingface_hub import HfApi, HfFolder, upload_file
from datasets import load_dataset, Dataset, DatasetDict
import sys


# Đường dẫn chứa các file txt
# folder_path = "/home/truongnn/trung/project/synthetic_data/data/input/txt"
# output_csv = "structured_law_dataset.csv"

# System prompt (vai trò chuyên gia)
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

# Tạo file CSV
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
  
# make_csv('demo.csv', 'data/input/txt')
  
def check_csv(csv_file: str, index_to_check: int):
  csv.field_size_limit(sys.maxsize)  # ✅ Tăng giới hạn cho ô dữ liệu lớn

  with open(csv_file, mode="r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      rows = list(reader)

      if index_to_check < len(rows):
          row = rows[index_to_check]
          print(f"🔎 Dòng số {index_to_check}:")
          print(f"📌 Title:\n{row['title']}\n")
          print(f"📌 System:\n{row['system']}\n")
          print(f"📌 Human:\n{row['human']}\n")
          print(f"📌 Context (1000 ký tự đầu):\n{row['context'][:1000]}...\n")
      else:
          print(f"❌ Index {index_to_check} vượt quá số dòng trong CSV ({len(rows)})")

check_csv("demo.csv", 301)


def upload(csv_path, repo_id):
  dataset = Dataset.from_csv(csv_path)
  data = DatasetDict({'train': dataset})
  data.push_to_hub(repo_id=repo_id, max_shard_size='150MB')
  
# upload('demo.csv', 'trungnguyen2331/law_extract')

# Đường dẫn tới folder chứa các file txt


# Lấy danh sách tất cả file .txt
def check_txt_file(folder_path, error_log_path):
  txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
  error_count = 0

  # Tạo file log
  with open(error_log_path, "w", encoding="utf-8") as log_file:
      for filename in tqdm(txt_files, desc="🔍 Đang kiểm tra file", unit="file"):
          file_path = os.path.join(folder_path, filename)
          try:
              with open(file_path, "r", encoding="utf-8") as f:
                  content = f.read()
                  if "Đang tải văn bản..." in content:
                      os.remove(file_path)
                      log_file.write(f"{os.path.splitext(filename)[0]}\n")
                      error_count += 1
          except Exception as e:
              print(f"⚠️ Lỗi khi xử lý {filename}: {e}")

  print(f"\n✅ Xử lý hoàn tất. Tổng file lỗi bị xoá: {error_count}")
  print(f"📄 Danh sách lưu tại: {error_log_path}")
  
# folder_path = "data/input/txt"  # 👈 chỉnh lại đường dẫn thật
# error_log_path = "data/error_file.txt"
# check_txt_file(folder_path, error_log_path)

# Thông tư 01/2025/TT-BNNMT về 03 quy chuẩn kỹ thuật quốc gia về chất lượng môi trường xung quanh do Bộ trưởng Bộ Nông nghiệp và Môi trường ban hành
# Thông tư 01/2025/TT-BNNMT về 03 quy chuẩn kỹ thuật quốc gia về chất lượng môi trường xung quanh do Bộ trưởng Bộ Nông nghiệp và Môi trường ban hành