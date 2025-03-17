# Use a Python base image
FROM python:3.11.11-slim

RUN apt-get update && apt-get install -y gcc

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN python -m pip install --upgrade pip

RUN python -m pip install -r requirements.txt

RUN python -m spacy download en_core_web_sm

# Copy the application files
COPY . .

# Expose the port your Flask app runs on
EXPOSE 4010

# Run the Flask application
CMD ["python", "app.py"]