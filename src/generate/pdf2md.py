import os
import argparse
import logging
import warnings
from tqdm import tqdm
from multiprocessing import Process, Queue
from markitdown import MarkItDown

# Suppress pdfminer warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Convert single file in subprocess (to enable timeout)
def convert_single_file(input_path, queue):
    try:
        md = MarkItDown(enable_plugins=False)
        result = md.convert(input_path)
        content = getattr(result, "markdown_content", None) or result.text_content
        queue.put(content)
    except Exception as e:
        queue.put(f"__ERROR__: {e}")

def convert_folder(input_dir, output_dir, timeout_sec=60):
    os.makedirs(output_dir, exist_ok=True)
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning("❌ No PDF files found.")
        return

    for i, file_name in enumerate(tqdm(pdf_files, desc="Converting PDFs"), start=1):
        input_path = os.path.join(input_dir, file_name)
        output_filename = os.path.splitext(file_name)[0] + ".md"
        output_path = os.path.join(output_dir, output_filename)

        queue = Queue()
        process = Process(target=convert_single_file, args=(input_path, queue))
        process.start()
        process.join(timeout=timeout_sec)

        if process.is_alive():
            process.terminate()
            logger.warning(f"⏱️ Timeout after {timeout_sec}s: {file_name}")
            continue

        content = queue.get()
        if isinstance(content, str) and content.startswith("__ERROR__"):
            logger.error(f"❌ Error processing {file_name}: {content.replace('__ERROR__:', '').strip()}")
            continue

        if not content.strip():
            logger.warning(f"⚠️ Empty content extracted from: {file_name}")
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"✅ Saved: {output_filename}")

            
if __name__=="__main__":
    convert_folder('/home/truongnn/trung/project/synthetic_data/data/input/pdf', "/home/truongnn/trung/project/synthetic_data/data/output/markdown", 60)