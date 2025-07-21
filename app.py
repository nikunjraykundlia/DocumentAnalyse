import os
import logging
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
import pytesseract
from PIL import Image
import io
from pdf2image import convert_from_bytes
from classifier import DocumentClassifier
from werkzeug.middleware.proxy_fix import ProxyFix

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Add CORS headers manually
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize the document classifier
classifier = DocumentClassifier()

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
        
        # Process the file with OCR
        try:
            if file_extension == 'pdf':
                # Process PDF file
                file_data = file.read()
                logging.info("Processing PDF file...")
                
                # Convert PDF pages to images
                try:
                    images = convert_from_bytes(file_data, dpi=300, first_page=1, last_page=5)  # Limit to first 5 pages
                    if not images:
                        return jsonify({
                            'error': 'No pages could be extracted from the PDF file.'
                        }), 400
                    
                    # Extract text from all pages
                    extracted_text = ""
                    for i, image in enumerate(images):
                        logging.info(f"Processing PDF page {i+1}")
                        page_text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
                        if page_text.strip():
                            extracted_text += f"{page_text.strip()}\n"
                    
                    if not extracted_text.strip():
                        return jsonify({
                            'error': 'No text could be extracted from the PDF pages. Please ensure the PDF contains readable text.'
                        }), 400
                        
                except Exception as pdf_error:
                    logging.error(f"PDF processing error: {str(pdf_error)}")
                    return jsonify({
                        'error': f'Error processing PDF: {str(pdf_error)}. Please ensure the PDF is not corrupted and contains text.'
                    }), 400
                    
            else:
                # Process image file
                image = Image.open(io.BytesIO(file.read()))
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
                
                if not extracted_text.strip():
                    return jsonify({
                        'error': 'No text could be extracted from the image. Please ensure the image contains readable text.'
                    }), 400
                
        except Exception as e:
            logging.error(f"OCR processing error: {str(e)}")
            return jsonify({
                'error': f'Error processing image: {str(e)}'
            }), 500
        
        # Classify the document
        try:
            classification_label = classifier.classify(extracted_text)
            confidence_score = classifier.get_confidence_score(extracted_text, classification_label)
        except Exception as e:
            logging.error(f"Classification error: {str(e)}")
            classification_label = 'Unknown'
            confidence_score = 0.0
        
        # Validate the document based on its classification
        try:
            validation_errors = classifier.validate(classification_label, extracted_text)
        except Exception as e:
            logging.error(f"Validation error: {str(e)}")
            validation_errors = ['Error during validation process']
        
        # Prepare response
        response_data = {
            'text': extracted_text,
            'label': classification_label,
            'confidence': confidence_score,
            'validation_errors': validation_errors,
            'success': True
        }
        
        logging.info(f"Document processed successfully: {classification_label}")
        return jsonify(response_data)
        
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
