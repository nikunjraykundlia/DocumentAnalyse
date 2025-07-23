import os
import logging
import base64
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for, session
import pytesseract
from PIL import Image
import io
import pdfplumber
import fitz  # PyMuPDF
from dotenv import load_dotenv
import threading

# --- Global in-memory extracted text store ---
extracted_text_store = {}
store_lock = threading.Lock()

def store_extracted_text(session_id, doc_id, text):
    with store_lock:
        if session_id not in extracted_text_store:
            extracted_text_store[session_id] = {}
        extracted_text_store[session_id][doc_id] = text

def get_extracted_text(session_id, doc_id):
    with store_lock:
        return extracted_text_store.get(session_id, {}).get(doc_id)


# Load environment variables before anything else
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

from classifier import DocumentClassifier
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Configure for Replit environment
if os.environ.get('REPLIT_DB_URL'):
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize the document classifier
classifier = DocumentClassifier(use_ai=True, use_ollama=True)

# Log AI engine status
ollama_status = getattr(classifier, 'use_ollama', False)
openai_status = getattr(classifier, 'use_ai', False)
if ollama_status:
    logging.info("Ollama/Llama AI classification ENABLED.")
elif openai_status:
    logging.info("OpenAI AI classification ENABLED.")
else:
    logging.info("Only rule-based classification is available. No AI engine enabled.")


@app.route('/')
def index():
    """Serve the main document upload page."""
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'API is working'})

@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload_document():
    """
    Handle document upload, OCR processing, classification, and validation.
    """
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return '', 200
    # Use Flask session for session_id (fallback to remote addr if not set)
    session_id = session.get('session_id')
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session['session_id'] = session_id
    
    try:
        # Debug logging
        logging.info(f"Upload request received. Method: {request.method}")
        logging.info(f"Files in request: {list(request.files.keys())}")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded', 'success': False}), 400
        
        file = request.files['file']
        logging.info(f"File uploaded: {file.filename}")
        
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
        if file.filename and '.' in file.filename:
            file_extension = file.filename.rsplit('.', 1)[1].lower()
        else:
            file_extension = ''
        
        if file_extension not in allowed_extensions:
            return jsonify({
                'error': f'Unsupported file type. Allowed: {", ".join(allowed_extensions)}',
                'success': False
            }), 400
        
        file_data = file.read()
        file.seek(0) # Reset file pointer for any subsequent reads

        # --- Image Preview Generation ---
        image_preview_b64 = None
        try:
            logging.info(f"Starting image preview generation for {file.filename}...")
            if file_extension == 'pdf':
                try:
                    pdf_document = fitz.open(stream=file_data, filetype="pdf")
                    if pdf_document.page_count > 0:
                        page = pdf_document.load_page(0)
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("jpeg")
                        
                        # Resize the image
                        image = Image.open(io.BytesIO(img_data))
                        image.thumbnail((300, 300))
                        buffered = io.BytesIO()
                        image.save(buffered, format="JPEG")
                        image_preview_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                        logging.info("PDF preview image generated successfully with PyMuPDF.")
                except Exception as pdf_exc:
                    logging.error(f"PDF preview generation failed for {file.filename}: {str(pdf_exc)}", exc_info=True)
            else:  # For image files
                image = Image.open(io.BytesIO(file_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image.thumbnail((300, 300))
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG")
                image_preview_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                logging.info("Image preview generated successfully for image file.")
        except Exception as img_exc:
            logging.error(f"Overall image preview generation failed for {file.filename}: {str(img_exc)}", exc_info=True)

        # --- Text Extraction ---
        try:
            if file_extension == 'pdf':
                pdf_bytes = file_data
            else:
                # Convert image to PDF in memory
                image = Image.open(io.BytesIO(file_data))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                pdf_buffer = io.BytesIO()
                image.save(pdf_buffer, format="PDF")
                pdf_bytes = pdf_buffer.getvalue()
            # Now always use PDF OCR pipeline
            extracted_text = ""
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        extracted_text += page_text + "\n"
            extracted_text = extracted_text.strip()
            if not extracted_text:
                logging.warning("pdfplumber found no text, PDF might be image-based.")
                # Fallback: Tesseract on PDF (PyMuPDF rasterize)
                try:
                    pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
                    for page in pdf_document:
                        pix = page.get_pixmap()
                        img_data = pix.tobytes("jpeg")
                        img = Image.open(io.BytesIO(img_data))
                        text = pytesseract.image_to_string(img, lang='eng+hin+guj', config='--oem 3 --psm 6')
                        if text and text.strip():
                            extracted_text += text + "\n"
                    extracted_text = extracted_text.strip()
                except Exception as tesseract_pdf_exc:
                    logging.warning(f"Tesseract PDF fallback failed: {tesseract_pdf_exc}")
            # OCR confidence/length check
            if not extracted_text or len(extracted_text.strip()) < 30:
                logging.warning("OCR output is too short or empty. Likely low-quality or noisy image.")
                extracted_text = "[WARNING] OCR failed: Image quality too low for reliable extraction. Please upload a clearer scan or photo."

            if not extracted_text.strip():
                logging.warning("No text could be extracted from the document.")

        except Exception as e:
            logging.error(f"Text extraction error: {str(e)}")
            return jsonify({
                'label': 'Error',
                'confidence': 0,
                'text': '[ERROR] Failed to extract text. Please check the file format and try again.'
            }), 200
        
        # Classify the document with language support
        try:
            # Detect language (basic unicode-based, can be replaced with langdetect if needed)
            def detect_language(text):
                guj_count = sum(0x0A80 <= ord(c) <= 0x0AFF for c in text)
                hin_count = sum(0x0900 <= ord(c) <= 0x097F for c in text)
                total = len(text)
                if total == 0:
                    return 'en'
                if guj_count / total > 0.3:
                    return 'guj'
                if hin_count / total > 0.3:
                    return 'hin'
                return 'en'

            lang_hint = detect_language(extracted_text)
            # Prepend language hint for Llama/Ollama
            prompt_context = extracted_text
            if lang_hint == 'guj':
                prompt_context = '[The following document is in Gujarati.]\n' + extracted_text
            elif lang_hint == 'hin':
                prompt_context = '[The following document is in Hindi.]\n' + extracted_text
            classification_label = classifier.classify(prompt_context)
            confidence_score = classifier.get_confidence_score(prompt_context, classification_label)
            # Only use 'Unidentified Document' if no label could be determined at all
            if not classification_label or classification_label.strip().lower() in ["", "unknown", "none"]:
                classification_label = 'Unidentified Document'
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            classification_label = 'Unknown'
            confidence_score = 0.0
        
        # Return results
        return jsonify({
            'label': classification_label,
            'confidence': confidence_score,
            'text': extracted_text,
            'image_preview': image_preview_b64,
            'success': True
        })

    except Exception as e:
        logging.error(f"Unexpected error in upload_document: {str(e)}")
        return jsonify({
            'error': f'An unexpected error occurred: {str(e)}',
            'success': False
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({
        'error': 'File too large. Please upload a smaller file.',
        'success': False
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors."""
    return jsonify({
        'error': 'Internal server error. Please try again.',
        'success': False
    }), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chatbot questions about all uploaded documents for the session.
    """
    import re
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Invalid request. Missing question.'}), 400

    question = data['question']
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session not found.'}), 400
    # Aggregate all extracted texts for this session
    with store_lock:
        context = "\n\n".join(extracted_text_store.get(session_id, {}).values())
    if not context:
        return jsonify({'error': 'No documents found for this session.'}), 404
    try:
        answer = classifier.answer_question(question, context)
        # Improved cleanup: remove any mix of leading/trailing * or ** before a colon in headings
        answer = re.sub(r'[\*]+([A-Za-z0-9 ,\-/]+):[\*]+', r'\1:', answer)
        answer = re.sub(r'^[\*]+([A-Za-z0-9 ,\-/]+):[\*]*', r'\1:', answer, flags=re.MULTILINE)
        return jsonify({'answer': answer})
    except Exception as e:
        logging.error(f"Chat endpoint error: {str(e)}")
        return jsonify({'error': 'An error occurred while getting the answer.'}), 500
