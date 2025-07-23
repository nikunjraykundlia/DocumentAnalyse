FROM python:3.10-slim

# Install system dependencies for OCR and PDF processing
RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt || pip install --no-cache-dir flask pytesseract pillow pdfplumber pdf2image python-dotenv openai scikit-learn ollama

CMD ["python", "main.py"]
