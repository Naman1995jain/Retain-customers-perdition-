# Retail Customer Churn Prediction

This project implements a machine learning solution to predict customer churn in a retail environment. It uses XGBoost algorithm to identify customers who are likely to stop doing business with the company, enabling proactive retention strategies.

## Project Structure

- `churn_predictor.py`: Main Python script containing the churn prediction logic
- `churn_xgb_pipeline.joblib`: Saved XGBoost model pipeline
- `Retail_data_analysis.ipynb`: Jupyter notebook with detailed data analysis and model development
- `requirements.txt`: List of Python dependencies
- `Dockerfile`: Container configuration for deploying the model

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Naman1995jain/Retain-customers-perdition-.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Using Jupyter Notebook
The `Retail_data_analysis.ipynb` notebook contains detailed analysis and step-by-step model development. To run it:
```bash
jupyter notebook Retail_data_analysis.ipynb
```

### Using the Python Script
To make predictions using the saved model:
```bash
python churn_predictor.py
```

### Docker Deployment
Build and run the Docker container:
```bash
docker build -t retail-churn-predictor .
docker run retail-churn-predictor
```

## Model Information

The project uses XGBoost (Extreme Gradient Boosting) algorithm for prediction, which is particularly effective for structured/tabular data. The model pipeline is saved in `churn_xgb_pipeline.joblib` for easy reuse and deployment..

## Author

- Naman Jain
