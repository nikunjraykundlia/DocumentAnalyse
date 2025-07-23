## 📃 Document Analyser: AI-Powered Document Classification

Document Analyser is a powerful web application designed to automatically classify and validate various types of documents uploaded as images or PDFs. The system now prioritizes **Llama-based AI classification via Ollama** as its primary engine, with OpenAI (GPT-4o) as an alternative and a robust rule-based fallback for unmatched cases. The pipeline includes Optical Character Recognition (OCR), direct text extraction, and multi-layered classification for fast and accurate results.

## ✨ Features

- **Llama (Ollama) AI Classification [Primary]**: Utilizes local Llama models (via Ollama) for advanced, privacy-friendly document classification. No internet required for AI processing.
- **OpenAI GPT-4o Integration [Optional]**: Optionally leverages OpenAI's GPT-4o for cloud-based AI classification if enabled and API key is provided.
- **Rule-Based Fallback**: A robust keyword-matching engine for documents that can't be classified by AI models.
- **Multi-Format Upload**: Supports PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, and more.
- **Advanced Text Extraction**: Uses `pdfplumber` for text-based PDFs and `Tesseract` OCR for images or scanned documents.
- **Confidence Scoring**: Provides a confidence score for each classification, helping to identify and flag uncertain results.
- **Document Validation**: Checks for required fields based on the classified document type.
- **Optional ML Model**: Includes a script (`model_train.py`) to train a custom `scikit-learn` model on your own data.
- **User-Friendly Interface**: Clean, modern UI for easy uploading and viewing of results.

## 🛠️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

You need to have the following installed on your system:

- **Python 3.8+**
- **Tesseract OCR Engine**: [Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- **Poppler**: PDF rendering for Windows: [Installation for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- **Ollama**: For Llama-based AI classification. [Ollama Installation](https://ollama.com/download)
- (Optional) **OpenAI API Key**: For cloud-based classification.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd DocumentAnalyse
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install flask pytesseract pillow pdfplumber pdf2image python-dotenv openai scikit-learn ollama
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root directory and add:

    ```env
    SESSION_SECRET="your-strong-secret-key"
    # (Optional) OpenAI API Key for cloud AI
    OPENAI_API_KEY="your-openai-api-key"
    ```
    
    Ollama runs locally and does not require an API key by default.

## 🚀 Usage

1. **Start Ollama** (for Llama AI):
   - Make sure the Ollama service is running on your machine (`ollama serve`).
   - Download and prepare your desired Llama model (e.g., `ollama run llama2`).

2. **Run the Flask app:**
   ```bash
   flask run
   ```

3. **Upload your documents via the web UI.**

- The system will use Llama/Ollama for classification by default.
- If Ollama is not available, and OpenAI is configured, it will use OpenAI for classification.
- If neither AI is available, the rule-based engine will classify documents using keyword matching.

Results, extracted text, and classification details will be displayed in the UI. The chatbot icon appears after processing is complete, allowing you to interact with the extracted content.

## 🤖 Classification Workflow

1. **Llama/Ollama (local AI)**: First priority for classification.
2. **OpenAI (cloud AI)**: Used if enabled and Ollama is unavailable.
3. **Rule-Based**: Fallback for unmatched or ambiguous cases.

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please create an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

1.  **Run the web application:**
    ```bash
    python app.py
    ```
    The application will be available at `http://127.0.0.1:5000`.

2.  **Use the web interface:**
    - Open your browser and navigate to `http://127.0.0.1:5000`.
    - Upload a document using the file input field.
    - The application will process the document and display the extracted text, classification label, confidence score, and any validation errors.

3.  **(Optional) Train the ML Model:**
    - Prepare your training data in a CSV file named `training_data.csv` with `text` and `label` columns.
    - Run the training script:
      ```bash
      python model_train.py
      ```
    - This will create a `model.pkl` file containing the trained model.

## ⚙️ Configuration

The application's behavior can be configured through the `.env` file and constants in the code.

- **AI Services**: In `app.py`, you can enable or disable AI features by changing the boolean flags in the `DocumentClassifier` initialization:
  ```python
  # To disable all AI, set both to False
  classifier = DocumentClassifier(use_ai=True, use_ollama=True)
  ```
- **Ollama Model**: The Ollama model can be configured in `classifier.py` inside the `_ollama_classify` method. The default is `llama3`.

## 🏗️ Project Structure

```
.DocumentAnalyse/
├── app.py              # Main Flask application, routes, and logic
├── classifier.py       # Document classification and validation logic
├── model_train.py      # Script to train the optional ML model
├── models.py           # Defines document types and their properties
├── templates/
│   └── index.html      # Frontend HTML template
├── .env                # Environment variables (you need to create this)
├── training_data.csv   # Sample data for model training
└── README.md           # This file
```

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to create an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
