# DocumentInsightHub: AI-Powered Document Classification

DocumentInsightHub is a powerful web application designed to automatically classify and validate various types of documents uploaded as images or PDFs. It leverages a sophisticated pipeline including Optical Character Recognition (OCR), direct text extraction, and a hybrid classification model to provide fast and accurate results.

## ✨ Features

- **Multi-Format Upload**: Supports a wide range of file types, including PDF, PNG, JPG, JPEG, GIF, BMP, and TIFF.
- **Advanced Text Extraction**: Uses `pdfplumber` for direct text extraction from text-based PDFs and `Tesseract` OCR for image-based documents.
- **Hybrid Classification System**:
  - **Rule-Based Engine**: A fast and robust primary classifier using extensive keyword matching.
  - **AI Enhancement**: Optionally uses local AI with **Ollama** or powerful cloud AI with **OpenAI (GPT-4o)** for documents that are difficult to classify with rules alone.
- **Confidence Scoring**: Provides a confidence score for each classification, helping to identify and flag uncertain results.
- **Document Validation**: Checks for the presence of required fields based on the classified document type.
- **Optional ML Model**: Includes a script (`model_train.py`) to train a custom `scikit-learn` classification model on your own data.
- **User-Friendly Interface**: A clean and modern UI for easy document uploading and viewing results.

## 🛠️ Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

You need to have the following installed on your system:

- **Python 3.8+**
- **Tesseract OCR Engine**: [Installation Guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
- **Poppler**: A PDF rendering library required for PDF processing. [Installation for Windows](https://github.com/oschwartz10612/poppler-windows/releases/)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd DocumentInsightHub
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    The project uses `uv` for package management, but you can install the packages with `pip` from a `requirements.txt` file. If one is not available, you can create it from `pyproject.toml` or install the main packages manually:
    ```bash
    pip install flask pytesseract pillow pdfplumber pdf2image python-dotenv openai scikit-learn ollama
    ```

4.  **Configure environment variables:**
    Create a file named `.env` in the root directory of the project and add the following variables:

    ```env
    # Flask secret key for session management
    SESSION_SECRET="your-strong-secret-key"

    # (Optional) OpenAI API Key for AI-enhanced classification
    OPENAI_API_KEY="your-openai-api-key"
    ```

## 🚀 Usage

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
.DocumentInsightHub/
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
