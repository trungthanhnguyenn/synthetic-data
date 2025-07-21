export PYTHONPATH=$(pwd)
uvicorn src.api.api_module:app --host 0.0.0.0 --port 8686 --reload
