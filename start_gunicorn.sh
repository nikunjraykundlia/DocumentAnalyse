#!/bin/bash
# Start Gunicorn with 4 workers for parallel file processing
exec gunicorn app:app --workers 4 --bind 0.0.0.0:10000
