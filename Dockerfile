FROM python:3.11-slim

# Install core build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code
COPY . .

# Expose port (Hugging Face Spaces default port is 7860)
EXPOSE 7860

# Configure environment variables
ENV PORT=7860
ENV FLASK_ENV=production

# Command to run the application
CMD ["python", "app.py"]
