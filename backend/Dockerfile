# Use a Python base image
FROM python:3.11.11-slim

RUN apt-get update && apt-get install -y gcc

# Set the working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Download NLTK corpora
RUN python -m nltk.downloader wordnet omw-1.4

# Download SpaCy model
RUN python -m spacy download en_core_web_sm

# Copy all app files
COPY . .

# Expose port
EXPOSE 4010

# Run app
CMD ["gunicorn", "-b", "0.0.0.0:4010", "app:app"]
