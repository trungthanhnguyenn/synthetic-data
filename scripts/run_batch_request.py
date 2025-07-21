# from openai import OpenAI
# from src.pipeline.batch_openai_processor import BatchOpenAIProcessor, BatchOpenAIConfig
# import csv
# import json

# def build_openai_batch_jsonl_from_csv(csv_path: str, jsonl_path: str, system_prompt: str) -> str:
#     with open(csv_path, "r", encoding="utf-8") as f_in, open(jsonl_path, "w", encoding="utf-8") as f_out:
#         reader = csv.DictReader(f_in)
#         for idx, row in enumerate(reader):
#             user_prompt = f"""📌 Title:
# {row['title'].strip()}

# 📌 Human:
# {row['human'].strip()}

# 📌 Context:
# {row['context'].strip()}"""

#             record = {
#                 "custom_id": f"req-{idx}",
#                 "method": "POST",
#                 "url": "/v1/chat/completions",
#                 "body": {
#                     "model": "gpt-4",
#                     "messages": [
#                         {"role": "system", "content": system_prompt.strip()},
#                         {"role": "user", "content": user_prompt}
#                     ],
#                     "temperature": 0.7,
#                     "top_p": 0.95,
#                     "max_tokens": 4096
#                 }
#             }
#             f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
#     return jsonl_path


# def run_openai_batch_pipeline(
#     csv_path: str,
#     jsonl_output_path: str,
#     system_prompt: str,
#     model_name: str = "gpt-4",
#     start: int = 0,
#     end: int = 10
# ):
#     # 1. Tạo file JSONL từ CSV
#     print("✅ Building batch input file...")
#     build_openai_batch_jsonl_from_csv(csv_path, jsonl_output_path, system_prompt)

#     # 2. Config
#     config = BatchOpenAIConfig(
#         model_name=model_name,
#         url="/v1/chat/completions",
#         dataset_name=None,  # not used here
#         num_samples_range=(start, end),
#         temperature=0.7,
#         top_p=0.95,
#         max_tokens=4096,
#         column_name_list=["prompt"],  # giả lập 1 key
#         system_prompt=system_prompt
#     )
#     config.input_file_path = jsonl_output_path

#     # 3. Client + Processor
#     client = OpenAI()
#     processor = BatchOpenAIProcessor(client, config)
    
#     # Gọi method từ class bạn đã viết
#     processor.generate_batch_response()

# if __name__ == "__main__":