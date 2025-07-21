#!/usr/bin/env python3
"""
Simple MVP Document Classification App
"""
import os
import io
import logging
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
import pytesseract
from PIL import Image
try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None
try:
    from classifier import DocumentClassifier
except ImportError:
    DocumentClassifier = None

# Configure logging
logging.basicConfig(level=logging.INFO)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-key-123")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# Initialize classifier
classifier = DocumentClassifier() if DocumentClassifier else None

# Simple HTML template
TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Classifier MVP</title>
    <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
    <style>
        .upload-area {
            border: 2px dashed #6c757d;
            padding: 2rem;
            text-align: center;
            border-radius: 0.5rem;
        }
        .result-card { margin-top: 1rem; }
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1 class="text-center mb-4">Document Classifier MVP</h1>
        
        <div class="row justify-content-center">
            <div class="col-md-8">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="upload-area mb-3">
                        <input type="file" name="file" id="fileInput" accept="image/*,.pdf" class="form-control mb-2">
                        <button type="submit" class="btn btn-primary" id="submitBtn">
                            <span id="btnText">Classify Document</span>
                            <span id="btnLoading" class="d-none">Processing...</span>
                        </button>
                    </div>
                </form>
                
                <div id="results" class="result-card d-none">
                    <div class="card">
                        <div class="card-body">
                            <h5>Results</h5>
                            <div id="resultContent"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('fileInput');
            const submitBtn = document.getElementById('submitBtn');
            const btnText = document.getElementById('btnText');
            const btnLoading = document.getElementById('btnLoading');
            const results = document.getElementById('results');
            const resultContent = document.getElementById('resultContent');
            
            if (!fileInput.files[0]) {
                alert('Please select a file');
                return;
            }
            
            // Show loading state
            btnText.classList.add('d-none');
            btnLoading.classList.remove('d-none');
            submitBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            
            try {
                console.log('Uploading file:', fileInput.files[0].name);
                
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                console.log('Response status:', response.status);
                const result = await response.json();
                console.log('Response data:', result);
                
                if (response.ok) {
                    resultContent.innerHTML = `
                        <div class="mb-2"><strong>Document Type:</strong> <span class="badge bg-primary">${result.label}</span></div>
                        <div class="mb-2"><strong>Confidence:</strong> ${Math.round(result.confidence * 100)}%</div>
                        <div class="mb-2"><strong>Validation:</strong> 
                            ${result.validation_errors.length === 0 ? 
                                '<span class="badge bg-success">Valid</span>' : 
                                '<span class="badge bg-warning">Issues Found</span>'}
                        </div>
                        ${result.validation_errors.length > 0 ? 
                            '<div class="mb-2"><small>Issues: ' + result.validation_errors.join(', ') + '</small></div>' : ''}
                        <div><strong>Extracted Text:</strong></div>
                        <div class="border p-2 mt-1" style="max-height: 200px; overflow-y: auto; font-family: monospace; font-size: 0.9rem;">${result.text || 'No text extracted'}</div>
                    `;
                    results.classList.remove('d-none');
                } else {
                    resultContent.innerHTML = `<div class="alert alert-danger">${result.error || 'An error occurred'}</div>`;
                    results.classList.remove('d-none');
                }
            } catch (error) {
                console.error('Upload error:', error);
                resultContent.innerHTML = `<div class="alert alert-danger">Network error: ${error.message}</div>`;
                results.classList.remove('d-none');
            } finally {
                // Reset button
                btnText.classList.remove('d-none');
                btnLoading.classList.add('d-none');
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(TEMPLATE)

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'MVP working'})

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_api():
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        logging.info("Upload request received")
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get file extension
        if file.filename and '.' in file.filename:
            file_extension = file.filename.rsplit('.', 1)[1].lower()
        else:
            file_extension = ''
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'pdf'}
        
        if file_extension not in allowed_extensions:
            return jsonify({'error': f'Unsupported file type: {file_extension}'}), 400
        
        logging.info(f"Processing {file_extension} file: {file.filename}")
        
        # Extract text using OCR
        extracted_text = ""
        
        try:
            if file_extension == 'pdf':
                # Process PDF
                if not convert_from_bytes:
                    return jsonify({'error': 'PDF processing not available'}), 400
                
                file_data = file.read()
                images = convert_from_bytes(file_data, dpi=200, first_page=1, last_page=2)
                
                for i, image in enumerate(images):
                    page_text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
                    if page_text.strip():
                        extracted_text += page_text.strip() + "\n"
            else:
                # Process image
                image = Image.open(io.BytesIO(file.read()))
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                extracted_text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
            
            extracted_text = extracted_text.strip()
            logging.info(f"Extracted {len(extracted_text)} characters of text")
            
            if not extracted_text:
                return jsonify({
                    'label': 'Unknown',
                    'confidence': 0.0,
                    'text': '',
                    'validation_errors': ['No text could be extracted from the document'],
                    'success': True
                })
            
            # Classify document
            if classifier:
                classification_label = classifier.classify(extracted_text)
                confidence = classifier.get_confidence_score(extracted_text, classification_label)
                validation_errors = classifier.validate(classification_label, extracted_text)
                
                result = {
                    'label': classification_label,
                    'confidence': confidence,
                    'validation_errors': validation_errors
                }
            else:
                result = {
                    'label': 'Unknown',
                    'confidence': 0.0,
                    'validation_errors': ['Classifier not available']
                }
            
            logging.info(f"Classification result: {result['label']} (confidence: {result['confidence']:.2f})")
            
            return jsonify({
                'label': result['label'],
                'confidence': result['confidence'],
                'text': extracted_text,
                'validation_errors': result['validation_errors'],
                'success': True
            })
            
        except Exception as ocr_error:
            logging.error(f"OCR error: {str(ocr_error)}")
            return jsonify({
                'error': f'OCR processing failed: {str(ocr_error)}',
                'success': False
            }), 400
            
    except Exception as e:
        logging.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)