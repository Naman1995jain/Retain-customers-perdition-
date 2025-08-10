# Use Python 3.9 slim image as the base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install -r requirements.txt

# Copy the application files into the container
COPY churn_predictor.py .
COPY churn_xgb_pipeline.joblib .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Command to run the Streamlit application
CMD ["streamlit", "run", "churn_predictor.py"]
