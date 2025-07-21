import os
import requests
import json
import re
from datasets import load_dataset
from dotenv import load_dotenv
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class SingleRequestConfig:
    api_url: str
    model_name: str
    router_name: str
    dataset_name: str
    temperature: float
    top_p: float
    max_tokens: int
    column_name: str  # context
    title_column: str  # title
    system_prompt: str
    output_dir: str


class SingleRequestProcessor:
    def __init__(self, config: SingleRequestConfig):
        self.config = config
        self.dataset = load_dataset(config.dataset_name, split="train")
        os.makedirs(config.output_dir, exist_ok=True)

    def safe_filename(self, filename: str) -> str:
        # Loại bỏ ký tự không hợp lệ trong tên file (nhưng không rút gọn nội dung)
        filename = re.sub(r'[\\/*?:"<>|]', '_', filename)
        return filename.strip()

    def process_all(self, start_idx=0, end_idx=1):
        for i in tqdm(range(start_idx, end_idx), desc="⏳ Processing requests"):
            item = self.dataset[i]
            context = item[self.config.column_name]
            title = item[self.config.title_column]
            safe_title = self.safe_filename(title)
            filename = os.path.join(self.config.output_dir, f"{safe_title}.json")

            full_prompt = f"{self.config.system_prompt.strip()}\n\nContext:\n{context.strip()}"
            payload = {
                "chat": full_prompt,
                "model_name": self.config.model_name,
                "router_name": self.config.router_name,
                "config": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "max_tokens": self.config.max_tokens,
                    "stream": False,
                    "get_thinking": False
                }
            }

            try:
                response = requests.post(self.config.api_url, json=payload)
                result = response.json()
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"[{i}] Saved to: {filename}")
            except Exception as e:
                print(f"[{i}] Error while saving {filename}: {e}")


if __name__ == "__main__":
    load_dotenv()

    config = SingleRequestConfig(
        api_url="http://localhost:8686/chat",
        model_name="deepseek-chat",
        router_name="deepseek",
        dataset_name="trungnguyen2331/law_extract",
        temperature=0.7,
        top_p=0.95,
        max_tokens=10000,
        column_name="context",
        title_column="title",
        system_prompt=r"""
Bạn là một chuyên gia pháp luật có nhiệm vụ trích xuất thông tin có cấu trúc từ văn bản pháp luật đã được số hóa (OCR hoặc văn bản thuần).

Bạn sẽ được cung cấp nội dung văn bản trong biến `context`.

Hãy thực hiện:

1. Nếu văn bản chứa đủ thông tin, hãy **chỉ trả về đối tượng JSON** sau (không thêm mô tả, không giải thích, không lý luận).
2. Nếu văn bản thiếu quá nhiều trường thông tin, hãy chỉ trả về đúng dòng: 
    `PDF không chứa đủ thông tin để điền vào bảng.`

**Yêu cầu định dạng kết quả:**
- Trả về JSON thuần (object), **không bọc trong chuỗi**, không chèn `<think>` hoặc bất kỳ dòng nào khác.
- Không thêm giải thích, tóm tắt.
- Tất cả các ngày phải theo định dạng `YYYY-MM-DD`
- Trường `"noi_dung"` phải chứa **nguyên vẹn toàn bộ nội dung văn bản từ phần "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" trở xuống** – không rút gọn, không viết tắt bằng `...`.

---

Cấu trúc JSON yêu cầu:
```json
{
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
  "noi_dung": "...",                // Toàn bộ nội dung của văn bản, không tóm tắt
  "linh_vuc": "..."                 // Lĩnh vực (VD: Giáo dục, Y tế)
}
"Hãy trích xuất thông tin theo yêu cầu."
""",
        output_dir="./law_outputs"
    )

    processor = SingleRequestProcessor(config)
    processor.process_all(start_idx=1005, end_idx=2000)