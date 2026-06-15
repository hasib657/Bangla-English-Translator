# Use the official Hugging Face Spaces Docker image as base
FROM huggingface/transformers-pytorch-gpu:latest

# Set working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./main.py /code/main.py

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Start the API server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
