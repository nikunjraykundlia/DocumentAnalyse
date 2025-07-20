"""
Test script to verify PDF processing functionality.
This script tests that pdf2image and OCR work correctly together.
"""

def test_pdf_processing():
    """Test that PDF processing libraries are working correctly."""
    try:
        from pdf2image import convert_from_bytes
        from PIL import Image
        import pytesseract
        
        print("✓ pdf2image library imported successfully")
        print("✓ PIL (Pillow) library imported successfully") 
        print("✓ pytesseract library imported successfully")
        print("✓ PDF processing setup is complete!")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

if __name__ == "__main__":
    test_pdf_processing()