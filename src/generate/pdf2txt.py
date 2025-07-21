import subprocess
import os
from src.utils.utils import get_filename_without_ext
from tqdm import tqdm

def convert_pdf_to_text(input_pdf_dir: str, output_txt_dir: str):

    if not os.path.exists(input_pdf_dir):
        raise FileNotFoundError(f"Không tìm thấy file PDF: {input_pdf_dir}")

    try:
        pdf_files = [file for file in os.listdir(input_pdf_dir) if file.endswith(".pdf")]
        for file in tqdm(pdf_files, desc="Converting PDF to TXT"):
            input_pdf_path = os.path.join(input_pdf_dir, file)
            output_basename = get_filename_without_ext(input_pdf_path)
            output_txt_path = os.path.join(output_txt_dir, f"{output_basename}.txt")

            subprocess.run(
                ["pdftotext", "-layout", input_pdf_path, output_txt_path],
                check=True
            )
            print(f"Đã chuyển đổi PDF sang TXT: {output_txt_path}")
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi chạy pdftotext: {e}")
    except Exception as e:
        print(f"Lỗi khác: {e}")