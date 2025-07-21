import os 
import time 
import openai
import json
from typing import List
from datasets import load_dataset
from groq import Groq
from dataclasses import dataclass
# from together import Together
import datetime
from dotenv import load_dotenv

# Lấy đường dẫn thư mục hiện tại chứa script
current_dir = os.path.dirname(__file__)
env_path = os.path.join(current_dir, ".env")

load_dotenv(dotenv_path=env_path)

@dataclass
class BatchOpenAIConfig:
    model_name:str
    url:str
    dataset_name:str
    num_samples_range: tuple
    temperature: float
    top_p: float
    max_tokens: int
    column_name_list: List[str]
    system_prompt: str


class BatchProcessError(Exception):
    pass

class BatchOpenAIProcessor:

    """
    Args:
        client (OpenAI): The OpenAI client object.
        batch_openai_config (BatchOpenAIConfig): The configuration object for batch processing.
        
    Attributes:
        client (OpenAI): The OpenAI client object.
        batch_openai_config (BatchOpenAIConfig): The configuration object for batch processing.
        dataset (Dataset): The dataset object.
        sub_dataset (Dataset): The sub-dataset object.
        column_name_list (List[str]): The list of column names.
        system_prompt (str): The system prompt for the batch processing.
        url (str): The URL for the model endpoint.
        model_name (str): The name of the model.
        num_samples_range (tuple): The range of sample numbers for the batch processing.
        temperature (float): The temperature for the batch processing.
        top_p (float): The top_p for the batch processing.
        max_tokens (int): The max_tokens for the batch processing.

    """

    def __init__(self , client: Groq, batch_openai_config: BatchOpenAIConfig):

        self.client = client 
        self.batch_openai_config = batch_openai_config
        self.dataset = load_dataset(batch_openai_config.dataset_name, split="train")
        self.sub_dataset = self.dataset.select(range(batch_openai_config.num_samples_range[0], batch_openai_config.num_samples_range[1]))
        self.column_name_list = batch_openai_config.column_name_list

    def build_request(self, prompt_list, start_index, end_index):
        file_name = f'batch_input{start_index}_{end_index}.jsonl'
        with open(file_name, 'w', encoding='utf-8') as f:
            req_id = start_index * len(self.column_name_list)
            for item in prompt_list:
                for col in self.column_name_list:
                    prompt_text = f"{col}: {item[col]}"
                    record = {
                        "custom_id": f"req-{req_id}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.batch_openai_config.model_name,
                            "messages": [
                                {"role": "system", "content": self.batch_openai_config.system_prompt},
                                {"role": "user", "content": prompt_text}
                            ]
                        }
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    req_id += 1
        return file_name

    
    def make_json_list(self):
        prompt = []
        for item in self.sub_dataset:
            record = {}
            for col in self.column_name_list:
                record[col] = item[col]
            prompt.append(record)
        return prompt


    def generate_batch_response(self):
        # Step 1: Build request and write to input file
        prompt_list = self.make_json_list()
        start_idx, end_idx = self.batch_openai_config.num_samples_range
        input_file = self.build_request(prompt_list, start_index=start_idx, end_index=end_idx)
        self.batch_openai_config.input_file_path = input_file

        # Step 2: Upload input file
        with open(input_file, "rb") as f:
            upload_resp = self.client.files.create(file=f, purpose="batch")
        input_file_id = upload_resp.id
        print(f"Uploaded input file: {input_file} | File ID: {input_file_id}")

        # Step 3: Create batch
        batch_resp = self.client.batches.create(
            completion_window="24h",
            endpoint=self.batch_openai_config.url,
            input_file_id=input_file_id
        )
        batch_id = batch_resp.id
        print(f"Batch job created. Batch ID: {batch_id}")

        # Step 4: Save metadata
        meta = {
            "batch_id": batch_id,
            "input_file_id": input_file_id,
            "input_file_path": input_file,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "status": "submitted"
        }
        meta_file = f"batch_meta_{batch_id}.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"Batch metadata saved to {meta_file}")

        # Step 5: Polling batch status
        print("Polling batch status...")
        status = batch_resp.status
        while status in ("validating", "in_progress", "finalizing"):
            print(f"Current status: {status}")
            time.sleep(5)
            batch_resp = self.client.batches.retrieve(batch_id)
            status = batch_resp.status

        print(f"Final status: {status}")

        # Step 6: Handle failure
        if status != "completed":
            print(f"Batch failed with status: {status}")
            if batch_resp.error_file_id:
                
                error_file_id = batch_resp.error_file_id
                print("Downloading error file...")
                error_content = self.client.files.content(error_file_id)
                error_path = f"batch_errors_{start_idx}_{end_idx}.jsonl"
                error_content.write_to_file(error_path)
                print(f"Error file saved to {error_path}")
                with open(error_path, "r", encoding="utf-8") as f:
                    print("First error line:")
                    print(f.readline())
            return False

        # Step 7: Download output
        output_file_id = batch_resp.output_file_id
        if not output_file_id:
            print("Batch completed but no output_file_id was returned.")
            return False

        print("Downloading result file...")
        output_jsonl_path = f"batch_results_{start_idx}_{end_idx}.jsonl"
        file_content = self.client.files.content(output_file_id)
        file_content.write_to_file(output_jsonl_path)
        print(f"Result file saved to {output_jsonl_path}")

        # Step 8: Convert to JSON
        self.merge_jsonl_files(output_jsonl_path)
        converted_json_path = f"converted_batch_results_{start_idx}_{end_idx}.json"

        # Step 9: Merge input and response
        self.merge_data(
            input_jsonl_path=input_file,
            response_json_path=converted_json_path,
            keys=self.batch_openai_config.column_name_list
        )

        print("Batch processing completed successfully.")


    def merge_jsonl_files(self, input_json_path:str ):
        data = []
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    data.append(obj)
                except Exception as e:
                    print("Not a valid JSON line:", line)   
                
        
        output_name = 'converted_' + os.path.splitext(input_json_path)[0] + '.json'
        output_path = os.path.join(os.path.dirname(input_json_path), output_name)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(data, out_f, ensure_ascii=False, indent=2)

        print(f"Saving file: {output_path}")

    def merge_data(
        self,
        input_jsonl_path: str,
        response_json_path: str,
        keys: List[str],
        output_dir: str = None
    ):
        """
        Merge grouped request input data with model-generated responses based on `custom_id`.

        Args:
            input_jsonl_path (str): Path to the .jsonl input file. Each group of N lines corresponds to a single logical unit.
            response_json_path (str): Path to the JSON file containing responses from the model, each linked by custom_id.
            keys (List[str]): List of logical keys corresponding to each line in a group (e.g. ["Question", "Reasoning", "Answer"]).
            output_dir (str, optional): Directory to save the merged output file. Defaults to the input file's directory.

        Output:
            A single JSON file with merged input and translated responses for each group.
        """
        group_size = len(keys)

        # Load input .jsonl file (each group_size lines = 1 logical unit)
        with open(input_jsonl_path, "r", encoding="utf-8") as f:
            input_entries = [json.loads(line) for line in f]

        # Load response JSON (structured as a list of responses, each with custom_id)
        with open(response_json_path, "r", encoding="utf-8") as f:
            response_data = json.load(f)

        # Group input entries into logical units
        grouped_inputs = []
        for i in range(0, len(input_entries), group_size):
            group = {}
            for j, key in enumerate(keys):
                group[key] = input_entries[i + j]["body"]["messages"][-1]["content"]
            group["custom_id"] = input_entries[i]["custom_id"]
            grouped_inputs.append(group)

        # Create lookup map from custom_id to response text
        response_lookup = {
            item["custom_id"]: item["response"]["body"]["choices"][0]["message"]["content"]
            for item in response_data
        }

        # Merge input group with corresponding responses
        merged_result = []
        for i in range(0, len(response_data), group_size):
            ids = [response_data[i + j]["custom_id"] for j in range(group_size)]
            matched = next((x for x in grouped_inputs if x["custom_id"] == ids[0]), None)

            if matched:
                for j, key in enumerate(keys):
                    matched[f"{key}_response"] = response_lookup.get(ids[j], "")
                merged_result.append(matched)

        # Determine output path
        if output_dir is None:
            output_dir = os.path.dirname(input_jsonl_path)

        output_name = f"merged_output_{os.path.basename(input_jsonl_path).replace('.jsonl', '')}.json"
        output_path = os.path.join(output_dir, output_name)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_result, f, ensure_ascii=False, indent=2)

        print(f"Merged file saved to: {output_path}")

# if __name__ == "__main__":
#     # Example usage
#     print(os.getenv("GROQ_API_KEY"))
#     client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#     config = BatchOpenAIConfig(
#         model_name="llama-3.3-70b-versatile",
#         url="/v1/chat/completions",
#         dataset_name="trungnguyen2331/law_extract",
#         num_samples_range=(2001, 2002),
#         temperature=0.7,
#         top_p=0.95,
#         max_tokens=10000,
#         column_name_list=['context'],
#         system_prompt=r"""
# Bạn là một chuyên gia pháp luật có nhiệm vụ trích xuất thông tin có cấu trúc từ văn bản pháp luật đã được số hóa (OCR hoặc định dạng văn bản thường).

# Yêu cầu:
# - Điền dữ liệu vào bảng JSON đúng định dạng.
# - Nếu thiếu quá nhiều thông tin quan trọng, trả về: PDF không chứa đủ thông tin để điền vào bảng.
# - Trả về JSON gốc, không bọc trong chuỗi.
# - Trường "noi_dung" phải chứa toàn bộ văn bản từ "Quốc hiệu Tiêu ngữ" trở xuống.

# Cấu trúc JSON yêu cầu:
# {
#   "so_hieu": "...",
#   "loai_vb": "...",
#   "noi_ban_hanh": "...",
#   "nguoi_ky": "...",
#   "ngay_ban_hanh": "...",
#   "ngay_hieu_luc": "...",
#   "ngay_cong_bao": "...",
#   "so_cong_bao": "...",
#   "tinh_trang": "...",
#   "tieu_de": "...",
#   "noi_dung": "...",
#   "linh_vuc": "..."
# }
# Hãy trích xuất thông tin theo yêu cầu.
# """
#     )

#     processor = BatchOpenAIProcessor(client=client, batch_openai_config=config)
#     try:
#         processor.generate_batch_response()
#     except Exception as e:
#         print(f"❌ Batch lỗi: {e}")
#         # logger.error(f"❌ Batch lỗi: {e}")
#         raise BatchProcessError(f"Batch processing failed: {e}")