from ..pipeline.gemini_endpoint.chat import GeminiChatPipeline
from ..pipeline.openai_endpoint.chat import OpenAIChatPipeline
from ..pipeline.openai_endpoint.chat import GroqChatPipeline
from src.pipeline.batch_processor.batch_groq_processor import BatchOpenAIProcessor, BatchOpenAIConfig, Groq
from ..utils.utils import get_all_env_values
from ..utils.loggers import logger
from ..utils.handle_response import handle_response
from ..pipeline.base_chat import BaseSettings, BaseConfig
from ..generate.pdf2json import generate_json
from ..generate.pdf2txt import convert_pdf_to_text
from tqdm import tqdm

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os
from dotenv import load_dotenv
# load_dotenv()
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # -> /home/truongnn/trung/project/LOHA
env_path = os.path.join(base_dir, ".env")

load_dotenv(dotenv_path=env_path)

import logging

def setup_logger():
    logger = logging.getLogger("model_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    # Ghi log ra console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Ghi log ra file
    file_handler = logging.FileHandler("model_api.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

logger = setup_logger()
app = FastAPI()

class FolderRequest(BaseModel):
    input_folder: str
    output_folder: str
    
class ChatRequest(BaseModel):
    chat: str
    model_name: str
    router_name: str
    config: dict | None = None

DEFAULT_SYSTEM_PROMPT = r"""
Bạn là một chuyên gia pháp luật có nhiệm vụ **trích xuất thông tin có cấu trúc** từ văn bản pháp luật đã được số hóa (OCR hoặc định dạng văn bản thường).

Yêu cầu bắt buộc:
- Dựa vào phần `context` bên dưới, hãy điền dữ liệu vào **bảng JSON** đúng theo định dạng và tên trường được chỉ định.
- Nếu **thiếu quá nhiều trường thông tin quan trọng**, hãy trả về chuỗi duy nhất:
  `PDF không chứa đủ thông tin để điền vào bảng.`
- Trả về đối tượng JSON **gốc, không bọc trong chuỗi**.
- Trường `"noi_dung"` phải chứa **toàn bộ nội dung văn bản từ phần **Quốc hiệu Tiêu ngữ trở xuống**, không được rút gọn hoặc mô tả bằng lời.
- Chỉ trả về đúng một trong hai:
  1. Một đối tượng JSON đầy đủ.
  2. Hoặc chuỗi `"PDF không chứa đủ thông tin để điền vào bảng."`

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
"""    
class BatchRequest(BaseModel):
    model_name: str
    url: str
    dataset_name: str
    num_samples_range: tuple
    temperature: float
    top_p: float
    max_tokens: int
    column_name_list: str
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    
PIPELINE_MAP = {
    "gemini": GeminiChatPipeline,      
    "groq": GroqChatPipeline,        
    "openai": OpenAIChatPipeline,
    "openrouter": OpenAIChatPipeline,
    "deepseek": OpenAIChatPipeline,
    "nvidia": OpenAIChatPipeline,
}

ROUTER_MAP = {
    'openai': 'OPENAI',
    'openrouter': 'OPENROUTER',
    'deepseek': 'DEEPSEEK',
    'groq': 'GROQ',
    'nvidia': 'NVIDIA',
    'gemini': 'GEMINI'}

@app.post("/chat")
async def chat_with_model(req: ChatRequest):

    """
    Args:
        req (ChatRequest): The request containing 
        chat (str): The chat message to send to the model.
        model_name (str): The name of the model to use.
        router_name (str): The name of the router to use.
    Returns:
        response (dict): The response from the model.
        status (str): The status of the response.
    """
    try:
        envs = get_all_env_values()
        model_name = req.model_name.lower()
        router_name = req.router_name.lower()
        pipeline_cls = PIPELINE_MAP.get(router_name)
        if not pipeline_cls:
            raise Exception("Model not supported")

        prefix = ROUTER_MAP.get(router_name)
        if not prefix:
            raise Exception("Router not supported")

        api_key = envs.get(f"{prefix}_KEY")
        model = envs.get(f"{prefix}_MODEL_NAME", model_name)
        base_url = envs.get(f"{prefix}_BASE_URL")
        settings = BaseSettings(
            model_name=model,
            base_url=base_url,
            api_key=api_key
        )
        user_config = req.config or {}
        config = BaseConfig(
            temperature=user_config.get("temperature", 0.6),
            top_p=user_config.get("top_p", 0.95),
            max_tokens=user_config.get("max_tokens", 4096),
            stream=user_config.get("stream", False),
            get_thinking=user_config.get("get_thinking", False),
        )

        pipeline = pipeline_cls(settings, config)

        response = await pipeline.send_messages_async(req.chat)
        return handle_response({"response": response}, "success")
    except Exception as e:
        return handle_response({"response": f"Error: Got {e}"}, "error")
    

@app.get("/check_model_status")
async def check_model_status():
    envs = get_all_env_values()
    status_dict = {}

    for router_name, prefix in ROUTER_MAP.items():
        try:
            logger.info(f"🔍 Checking router: {router_name} | Prefix: {prefix}")

            api_key = envs.get(f"{prefix}_KEY")
            model_name = envs.get(f"{prefix}_MODEL_NAME")
            base_url = envs.get(f"{prefix}_BASE_URL")

        
            pipeline_cls = PIPELINE_MAP.get(router_name)
            if not pipeline_cls:
                status_dict[router_name] = "pending: no pipeline"
                continue

            settings = BaseSettings(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key
            )

            config = BaseConfig(
                temperature=0.95,
                top_p=1.0,
                max_tokens=20,
                stream=False,
                get_thinking=False
            )

            pipeline = pipeline_cls(settings, config)
            await pipeline.send_messages_async("ping")

            status_dict[router_name] = "ready"
        except Exception as e:
            status_dict[router_name] = f"pending with error: {e}"

    return status_dict

@app.post("/generate_txt")
def generate_txt_folders(request: FolderRequest):
    input_pdf_dir = request.input_folder
    output_txt_dir = request.output_folder

    if not os.path.exists(input_pdf_dir):
        return {"error": f"Input folder {input_pdf_dir} does not exist."}
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir, exist_ok=True)

    convert_pdf_to_text(input_pdf_dir, output_txt_dir)
    return {"message": f"Converted all PDFs in {input_pdf_dir} to TXT in {output_txt_dir}"}

@app.post("/generate_json")
def generate_json_folders(request: FolderRequest):
    input_txt_dir = request.input_folder
    output_json_dir = request.output_folder

    if not os.path.exists(input_txt_dir):
        return {"error": f"Input folder {input_txt_dir} does not exist."}
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir, exist_ok=True)

    txt_files = [file for file in os.listdir(input_txt_dir) if file.endswith(".txt")]
    
    for file in tqdm(txt_files, desc="Converting TXT to JSON"):
        input_txt_path = os.path.join(input_txt_dir, file)
        generate_json(input_txt_path, output_json_dir)
    return {"message": f"Converted all TXTs in {input_txt_dir} to JSON in {output_json_dir}"}

@app.post("/generate_batch")
async def generate_batch(req: BatchRequest):
    try:
        
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        system_prompt = req.system_prompt or DEFAULT_SYSTEM_PROMPT
        config = BatchOpenAIConfig(
            model_name=req.model_name,
            url=req.url,
            dataset_name=req.dataset_name,
            num_samples_range=req.num_samples_range,
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            column_name_list=req.column_name_list.split(","),
            system_prompt=system_prompt
        )
        
        logger.info(f"Starting batch generation with config: {config}")
        processor = BatchOpenAIProcessor(client, config)
        processor.generate_batch_response()
        return print("status: completed")
    
    except ValueError as ve:
        logger.error(f" ValueError: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    
    except Exception as e:
        logger.error(f" Unexpected error in /generate_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
# @