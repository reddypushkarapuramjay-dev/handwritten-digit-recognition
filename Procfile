# Handwritten Digit Recognition - Process File for Deployment
# This file tells the platform (Render, Heroku, etc.) how to run the application

# Web process - runs the Flask app with Gunicorn
web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 app:app

# Alternative for development (uncomment for local testing)
# web: python app.py