# Use Python 3.11 as base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run the Streamlit application
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0"]
