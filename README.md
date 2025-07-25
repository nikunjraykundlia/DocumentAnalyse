## 📃 Document Analyse: AI-Powered Document Classification

Document Analyse is a powerful web application that automatically classifies, validates, and enables Q&A on a wide range of uploaded documents (PDFs, images, scans). It prioritizes **Llama-based AI classification via Ollama** for privacy-friendly, local inference, with OpenAI (GPT-4o) as an optional cloud fallback, and robust rule-based logic for unmatched cases. The system features advanced OCR, multi-format support, confidence scoring, and a context-aware chatbot for interactive document insights.

## ✨ Features

- **Llama (Ollama) AI Classification [Primary]**: Utilizes local Llama models (via Ollama) for advanced, privacy-friendly document classification. No internet required for AI processing.
- **OpenAI GPT-4o Integration [Optional]**: Optionally leverages OpenAI's GPT-4o for cloud-based AI classification if enabled and API key is provided.
- **Rule-Based Fallback**: Robust keyword-matching and field validation for documents not handled by AI models.
- **Multi-Format Upload**: Supports PDF, PNG, JPG, JPEG, GIF, BMP, TIFF, and more.
- **Advanced Text Extraction**: Uses `pdfplumber` for text-based PDFs, `Tesseract` OCR for images/scans, and PyMuPDF for PDF parsing.
- **Confidence Scoring**: Returns a confidence score for each classification to help flag uncertain results.
- **Document Validation**: Checks for required fields based on classified document type; highlights missing or invalid fields.
- **Custom ML Model Support**: Includes a script (`model_train.py`) to train a custom `scikit-learn` model on your own data.
- **Interactive Chatbot**: Built-in chatbot (Ollama-powered) answers questions about the extracted content of all uploaded documents in your session, strictly using only the visible document context.
- **Multi-File Upload & Processing**: Upload and process multiple documents in a single session; extracted text and results are session-scoped.
- **User-Friendly Interface**: Clean, modern UI for easy uploading, result viewing, and chatbot access (toggle appears after extraction).
- **Session-Based Context**: Keeps extracted text and chatbot context per user session for privacy and continuity.
- **Error Handling**: Handles large files, unsupported formats, and server errors gracefully with clear feedback.
- **CORS Support**: Configured for cross-origin requests, suitable for deployment as a web service.

## 🛠️ Getting Started

Follow these instructions to set up and run Document Analyse locally.

### Prerequisites
- **Python 3.8+**
- **Tesseract OCR Engine**: [Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- **Poppler**: PDF rendering for Windows: [Installation for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)
- **Ollama**: For Llama-based AI classification. [Ollama Installation](https://ollama.com/download)
- (Optional) **OpenAI API Key**: For cloud-based classification.

### Installation

1. **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd DocumentAnalyse
    ```
2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```
3. **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure environment variables:**
    Create a `.env` file in the root directory and add:
    ```env
    SESSION_SECRET="your-strong-secret-key"
    # (Optional) OpenAI API Key for cloud AI
    OPENAI_API_KEY="your-openai-api-key"
    ```

## ⚠️ Note on Cloud Deployments (Render, etc.)
- Ensure all environment variables are set in the Render dashboard.
- Set the start command to `gunicorn app:app` or `python app.py`.
- For Ollama, you must run the Ollama server on the same host (not available on most free cloud platforms).
- For OpenAI, provide your API key if you want to enable cloud classification.
- The app is CORS-enabled for web deployment.

## 🚀 Usage

1. **Start the application:**
    ```bash
    python app.py
    ```
    or
    ```bash
    gunicorn app:app
    ```
2. **Use the web interface:**
    - Open your browser and navigate to `http://127.0.0.1:5000`.
    - Upload one or more documents using the file input field.
    - The application will process each document, display extracted text, classification label, confidence score, and any validation errors.
    - Once extraction is complete, the chatbot icon appears (bottom right). Click it to ask questions about the uploaded documents.

## 📚 Supported Document Types (Examples)
- Electricity Bill
- Property Tax Bill
- Birth Certificate
- Mobile Phone Bill
- (Easily extensible via `models.py`)

## 🤖 Chatbot Q&A
- The chatbot answers questions strictly based on the extracted content of your uploaded documents, using Ollama (Llama 3) for local inference.
- If Ollama is unavailable, chat functionality is disabled.
- The chatbot does not use external knowledge and will state if the answer is not found in your documents.

## 🧩 Extending & Customizing
- Add new document types or validation rules in `models.py`.
- Train a custom ML model using `model_train.py` and integrate it in `classifier.py`.
- Adjust UI/UX in the frontend files (if present).

## 📝 License
MIT License. See [LICENSE](LICENSE) for details.

---

For questions or contributions, please open an issue or pull request!
- **Ollama/Llama-based AI classification is only available on local machines or servers where you can install and run Ollama.**
- **On Render or other cloud platforms, only OpenAI (if configured) or rule-based classification will be available.**
- If you require Llama/Ollama, run the app on your own machine or a compatible server.

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
