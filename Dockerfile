# Use Python 3.9 slim image as the base
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8000 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY churn_predictor.py .
COPY churn_xgb_pipeline.joblib .

# Expose the port that Streamlit runs on
EXPOSE 8000

# Command to run the Streamlit application
CMD ["streamlit", "run", "churn_predictor.py"]