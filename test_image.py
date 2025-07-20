"""
Create a simple test image with text for testing OCR functionality.
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a simple test image with electricity bill text."""
    # Create a white image
    width, height = 600, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to use a default font, fallback to PIL default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add electricity bill text
    text_lines = [
        "ELECTRICITY BILL",
        "",
        "Account Number: 123456789",
        "Service Address: 123 Main Street", 
        "Billing Period: Jan 1 - Jan 31, 2024",
        "",
        "Previous Reading: 12,345 kWh",
        "Current Reading: 12,795 kWh",
        "kWh Used: 450",
        "",
        "Amount Due: $85.50",
        "Due Date: February 15, 2024"
    ]
    
    y = 50
    for line in text_lines:
        if line.strip():  # Only draw non-empty lines
            draw.text((50, y), line, fill='black', font=font)
        y += 25
    
    filename = "test_electricity_bill.png"
    image.save(filename)
    print(f"Test image created: {filename}")
    return filename

if __name__ == "__main__":
    create_test_image()