# Document Classification System

## Overview

This is a Flask-based web application that performs OCR (Optical Character Recognition) on uploaded document images and PDF files, then classifies them into predefined categories. The system uses rule-based classification with optional AI enhancement through OpenAI's API. It supports various document types including electricity bills, property tax bills, birth certificates, mobile phone bills, water bills, gas bills, and Indian identity documents (PAN Card, Aadhaar Card).

## User Preferences

Preferred communication style: Simple, everyday language.
UI Design: Beautiful blue-white gradient theme with glassmorphism effects
Document Support: Enhanced classification for Indian documents (PAN Card, Aadhaar Card)

## Recent Changes

### 2025-07-22: Enhanced Document Classification & UI Redesign
- Fixed PDF text extraction using pdfplumber for direct text extraction
- Enhanced document classification with comprehensive keyword templates for all document types
- Added 25+ keywords per document type for stronger classification accuracy
- Improved scoring algorithm with normalized confidence thresholds
- Fixed double file selection popup issue with enhanced JavaScript handling
- Created beautiful blue gradient UI with glassmorphism design and white gradient headlines
- Enhanced ProxyFix configuration for better Replit domain connectivity
- Improved name detection validation with regex patterns for Indian documents

## System Architecture

### Frontend Architecture
- **Technology**: HTML templates with Bootstrap for UI styling
- **Framework**: Server-rendered templates using Flask's Jinja2 templating engine
- **Styling**: Bootstrap with dark theme and Font Awesome icons
- **Interaction**: Form-based file upload with JavaScript for enhanced UX (drag-and-drop, progress indicators)

### Backend Architecture
- **Framework**: Flask (Python web framework)
- **Design Pattern**: Simple MVC pattern with route handlers in `app.py`
- **Middleware**: ProxyFix for handling reverse proxy headers
- **OCR Processing**: Tesseract via pytesseract library for text extraction from images
- **Classification**: Custom rule-based classifier with optional OpenAI integration

## Key Components

### Document Processing Pipeline
1. **File Upload Handler** (`/upload` route): Validates file types (images and PDFs) and processes uploads
2. **PDF Converter**: Converts PDF pages to images using pdf2image library
3. **OCR Engine**: Extracts text from images and PDF pages using Tesseract
4. **Document Classifier**: Categorizes documents based on extracted text using rule-based and AI methods
5. **Validation Engine**: Validates required fields for each document type

### Document Classification System
- **Rule-based Classification**: Uses keyword matching and pattern recognition
- **AI Enhancement**: Optional OpenAI integration for improved accuracy
- **Document Types**: Defined in `models.py` with specific validation rules
- **Extensible Design**: Easy to add new document types and validation rules

### Core Modules
- **`app.py`**: Main Flask application with route handlers
- **`classifier.py`**: Document classification logic with AI integration
- **`models.py`**: Data models for document types and validation rules
- **`model_train.py`**: Machine learning training utilities (optional enhancement)

## Data Flow

1. **Upload**: User uploads document image through web interface
2. **Validation**: File type and size validation
3. **OCR Processing**: Text extraction using Tesseract
4. **Classification**: Rule-based matching with optional AI enhancement
5. **Validation**: Field extraction and validation based on document type
6. **Response**: JSON response with classification results and confidence scores

## External Dependencies

### Required Services
- **Tesseract OCR**: System-level OCR engine for text extraction
- **PIL/Pillow**: Image processing library
- **Poppler**: PDF rendering library (system dependency)
- **pdf2image**: Python library for converting PDF pages to images

### Optional Services
- **OpenAI API**: For AI-enhanced document classification
- **Environment Variables**: 
  - `OPENAI_API_KEY`: Optional API key for AI features
  - `SESSION_SECRET`: Flask session secret key

### Python Libraries
- Flask: Web framework
- pytesseract: Tesseract Python wrapper
- PIL: Image processing
- pdf2image: PDF to image conversion
- scikit-learn: Machine learning capabilities (for training custom models)
- OpenAI: AI integration (optional)

## Deployment Strategy

### Development
- **Environment**: Flask development server
- **Configuration**: Debug mode enabled
- **Host**: Configurable host/port (defaults to 0.0.0.0:5000)

### Production Considerations
- **WSGI**: Compatible with standard WSGI servers
- **Security**: Environment-based secret key management
- **Proxy Support**: ProxyFix middleware for reverse proxy deployments
- **Logging**: Configurable logging levels

### File Structure
- Static files served through Flask's static file handler
- Templates in `/templates` directory
- Modular design allows for easy scaling and maintenance

The application is designed to be lightweight and easily deployable while maintaining extensibility for adding new document types and classification methods.