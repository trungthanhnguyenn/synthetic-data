import pandas as pd
from typing import List, Optional
import json
import re
from tqdm import tqdm
from datetime import datetime
import unicodedata

class OutputLLMProcessor():
    def __init__(
        self,
        input_json: str,
        col_names: Optional[List[str]] = None,
        output_json: str = ""
    ):
        self.input_json = input_json
        self.col_names = col_names or ['context_response', 'custom_id']
        self.output_json = output_json

    def extract_llm_response(self):
        """
        Extract specified fields from a JSON list of records (not indexed) and save as list of records.
        """
        df = pd.read_json(self.input_json, orient='records')

        # Kiểm tra cột thiếu
        missing_cols = [col for col in self.col_names if col not in df.columns]
        if missing_cols:
            raise ValueError(f"⚠️ Missing columns in input: {missing_cols}")
        
        # Trích xuất các cột cần thiết
        ext = df[self.col_names]
        return ext
    
def safe_json_loads(s):
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        cleaned = re.sub(r"^```json\s*|\s*```$", "", s.strip(), flags=re.DOTALL)
        cleaned = cleaned.replace("\\", "")
        return json.loads(cleaned)
    except Exception as e:
        print("❌ Error parsing JSON:", e)
        return None

def safe_get(d, key):
    if isinstance(d, dict):
        return (d.get(key) or "").strip()
    return ""

def format_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")
    except Exception:
        return date_str
    
def extract_and_map_fields_from_df(llm_df: pd.DataFrame, csv_path: str, output_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []

    for _, item in tqdm(llm_df.iterrows(), total=len(llm_df), desc="🔍 Processing LLM responses"):
        context_json = safe_json_loads(item.get("context_response", ""))
        if context_json is None:
            print(f"⚠️ Skipping due to JSON error: custom_id={item.get('custom_id')}")
            continue

        custom_id = item.get("custom_id", "")
        try:
            index = int(custom_id.split("-")[1])
            matched_title = df.loc[index, "title"]
        except Exception as e:
            print(f"⚠️ Failed to get title for custom_id={custom_id}: {e}")
            matched_title = None

        record = {
            "Tên văn bản": matched_title,
            "Số hiệu": safe_get(context_json, "so_hieu").replace("\\", ""),
            "Loại văn bản": safe_get(context_json, "loai_vb"),
            "Lĩnh vực": safe_get(context_json, "linh_vuc"),
            "Nơi ban hành": safe_get(context_json, "noi_ban_hanh"),
            "Người ký": safe_get(context_json, "nguoi_ky"),
            "Ngày ban hành": format_date(safe_get(context_json, "ngay_ban_hanh")),
            "Ngày hiệu lực": format_date(safe_get(context_json, "ngay_hieu_luc")),
            "Ngày đăng": format_date(safe_get(context_json, "ngay_cong_bao")),
            "Số công báo": safe_get(context_json, "so_cong_bao"),
            "Tình trạng": safe_get(context_json, "tinh_trang"),
            "Nội dung": safe_get(context_json, "noi_dung"),
        }

        records.append(record)

    output_df = pd.DataFrame(records)

    if output_path:
        output_df.to_json(output_path, orient="records", indent=2, force_ascii=False)
        print(f"✅ Done. Output saved to {output_path}")
    else:
        print(f"✅ Done. {len(output_df)} records extracted (no file saved).")

    return output_df

def normalize(text):
    if not isinstance(text, str):
        return ""

    # Chuẩn hóa Unicode
    text = unicodedata.normalize("NFKC", text)

    # Thay thế dấu nháy và ngoặc đặc biệt
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = text.replace("–", "-").replace("—", "-")  # dash variants

    # Xoá ký tự không in được, kể cả dấu space đặc biệt
    text = ''.join(c for c in text if c.isprintable())
    text = text.replace('\u00A0', ' ')  # non-breaking space

    # Thay gạch dưới bằng gạch chéo
    text = text.replace('_', '/')

    # Xoá xuống dòng, tab, dấu ngoặc kép thừa
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = text.strip('"').strip("'")

    # Rút gọn khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()

def match_url_and_save(final_df: pd.DataFrame, url_json_path: str, output_csv_path: str):
    # 1. Normalize tiêu đề
    final_df["normalized_title"] = final_df["Tên văn bản"].apply(normalize)

    # 2. Load JSON chứa URL
    with open(url_json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 3. Tạo dict map normalized_title → url
    title_to_url = {
        normalize(item["title"]): item["url"]
        for item in json_data
        if "title" in item and "url" in item
    }

    # 4. Ánh xạ URL
    tqdm.pandas(desc="🔗 Matching URLs")
    final_df["url"] = final_df["normalized_title"].progress_apply(lambda x: title_to_url.get(x, ""))

    # 5. Báo thiếu
    missing = final_df["url"].eq("").sum()
    print(f"⚠️ Không tìm thấy URL cho {missing} văn bản.")

    # 6. Xoá cột phụ và lưu file
    final_df.drop(columns=["normalized_title"], inplace=True)
    final_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ File kết quả đã lưu tại: {output_csv_path}")
    
# if __name__ == "__main__":

#     # 1. Đường dẫn
#     llm_output_path = "/home/truongnn/trung/data/law/out_llm_json/merged_output_batch_input2002_2003.json"
#     raw_csv_path = "/home/truongnn/trung/build_synthetic_data/data/input/csv/demo.csv"
#     url_json_path = "/home/truongnn/trung/build_synthetic_data/data/input/json/VBPL_merged_all.json"
#     output_csv_path = "/home/truongnn/trung/build_synthetic_data/data/output/csv/demo.csv"

#     # 2. Trích xuất context_response và custom_id
#     processor = OutputLLMProcessor(llm_output_path)
#     llm_df = processor.extract_llm_response()

#     # 3. Parse JSON trong context, extract các trường cần
#     parsed_df = extract_and_map_fields_from_df(llm_df, raw_csv_path)

#     # 4. Ghép URL dựa trên normalize(Tên văn bản) <-> normalize(title)
#     match_url_and_save(parsed_df, url_json_path, output_csv_path)