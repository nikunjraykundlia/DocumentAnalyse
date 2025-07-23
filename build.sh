#!/usr/bin/env bash
# Render build script: install Tesseract OCR and Python dependencies
set -e
apt-get update
apt-get install -y tesseract-ocr
pip install -r requirements.txt
