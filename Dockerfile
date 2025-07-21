FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (poppler-utils for pdftotext)
RUN apt-get update && \
    apt-get install -y poppler-utils && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy all code
COPY . .

# Expose FastAPI port
EXPOSE 8686

# Run the FastAPI app
CMD ["uvicorn", "src.api.api_module:app", "--host", "0.0.0.0", "--port", "8686", "--reload"]