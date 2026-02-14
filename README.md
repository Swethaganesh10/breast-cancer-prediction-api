# Breast Cancer Prediction API

**Author:** Swetha Ganesh  
**Course:** IE7374 Machine Learning Operations - Spring 2026  
**Modified Lab:** FastAPI Lab 1

---

## Overview

This project is a modified version of the FastAPI Lab, implementing a REST API for breast cancer diagnosis prediction using machine learning. The API accepts tumor measurements and predicts whether a tumor is malignant (cancerous) or benign (non-cancerous).

---

## Modifications from Original Lab

This lab has been significantly modified from the original Iris classification example:

1. **Dataset Change**
   - **Original:** Iris flower dataset (150 samples, 4 features, 3 classes)
   - **Modified:** Breast Cancer Wisconsin dataset (569 samples, 30 features, 2 classes)

2. **Model Upgrade**
   - **Original:** Decision Tree Classifier
   - **Modified:** Random Forest Classifier with 100 trees
   - **Performance:** Achieved 95.6% accuracy on test set

3. **Enhanced Features**
   - Added confidence scores and probability distributions in predictions
   - Implemented comprehensive input validation with value constraints
   - Added `/model-info` endpoint to display model details
   - Enhanced response structure with human-readable diagnosis labels

4. **Input Validation**
   - Added Pydantic field validators with range constraints (e.g., >= 0, <= 1)
   - Improved error handling with descriptive error messages
   - Validates all 30 tumor measurement features

5. **API Enhancements**
   - Custom API title and description
   - Better documentation in endpoint descriptions
   - Response includes prediction, diagnosis label, confidence, and probabilities

---

## Dataset Information

**Name:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** scikit-learn built-in dataset  
**Samples:** 569 tumor samples  
**Features:** 30 numerical features computed from digitized images of breast mass

**Feature Categories:**
- **Mean values** (10 features): radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension
- **Standard error values** (10 features): SE of above measurements
- **Worst values** (10 features): Mean of the three largest values

**Target Classes:**
- **0:** Malignant (cancerous)
- **1:** Benign (non-cancerous)

---

## Project Structure
```
breast-cancer-api/
├── src/
│   ├── data.py           # Data loading and preprocessing
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction function
│   └── main.py           # FastAPI application
├── model/
│   └── breast_cancer_model.pkl  # Trained model
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Swethaganesh10/breast-cancer-prediction-api.git
cd breast-cancer-prediction-api
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Train the model** (optional - model is already included)
```bash
cd src
python train.py
```

4. **Start the API server**
```bash
cd src
uvicorn main:app --reload
```

The API will be available at: `http://localhost:8000`

---

## API Endpoints

### 1. Health Check
**GET /** 

Checks if the API is running and model is loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "api_version": "2.0.0",
  "author": "Swetha Ganesh"
}
```

### 2. Model Information
**GET /model-info**

Returns detailed information about the model and modifications.

**Response:**
```json
{
  "model_type": "Random Forest Classifier",
  "dataset": "Breast Cancer Wisconsin (Diagnostic)",
  "n_features": 30,
  "n_samples": 569,
  "classes": {
    "0": "Malignant (Cancerous)",
    "1": "Benign (Non-cancerous)"
  },
  "modifications": [...]
}
```

### 3. Predict Cancer
**POST /predict**

Predicts whether a tumor is malignant or benign based on measurements.

**Request Body:**
```json
{
  "mean_radius": 17.99,
  "mean_texture": 10.38,
  "mean_perimeter": 122.8,
  "mean_area": 1001,
  "mean_smoothness": 0.1184,
  "mean_compactness": 0.2776,
  "mean_concavity": 0.3001,
  "mean_concave_points": 0.1471,
  "mean_symmetry": 0.2419,
  "mean_fractal_dimension": 0.07871
}
```

**Response:**
```json
{
  "prediction": 1,
  "diagnosis": "Benign (Non-cancerous)",
  "confidence": 0.81,
  "probabilities": {
    "malignant": 0.19,
    "benign": 0.81
  }
}
```

---

## Testing the API

### Using the Interactive Documentation

1. Start the API server
2. Open your browser and navigate to: `http://localhost:8000/docs`
3. You'll see the Swagger UI with all endpoints
4. Click on any endpoint to expand it
5. Click "Try it out" to test with sample data
6. Click "Execute" to see the response

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99,
    "mean_texture": 10.38,
    "mean_perimeter": 122.8,
    "mean_area": 1001,
    "mean_smoothness": 0.1184,
    "mean_compactness": 0.2776,
    "mean_concavity": 0.3001,
    "mean_concave_points": 0.1471,
    "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871
  }'
```

---

## Model Performance

**Accuracy:** 95.61%

**Classification Report:**
```
              precision    recall  f1-score   support
   Malignant       0.95      0.93      0.94        42
      Benign       0.96      0.97      0.97        72
    accuracy                           0.96       114
```

**Confusion Matrix:**
```
[[39  3]
 [ 2 70]]
```

---

## Technologies Used

- **FastAPI** - Modern web framework for building APIs
- **Uvicorn** - ASGI server for running FastAPI
- **scikit-learn** - Machine learning library for model training
- **Pydantic** - Data validation using Python type hints
- **NumPy** - Numerical computing
- **Joblib** - Model serialization

---

## Key Learning Outcomes

1. **ML Model Deployment** - Learned how to serve machine learning models as REST APIs
2. **API Development** - Built production-ready API endpoints with proper validation
3. **Data Validation** - Implemented input validation using Pydantic
4. **Model Evaluation** - Evaluated and reported model performance metrics
5. **Documentation** - Created comprehensive API documentation

---

## Future Enhancements

Potential improvements for this project:
- Add batch prediction endpoint for multiple samples
- Implement model versioning and A/B testing
- Add authentication and rate limiting
- Deploy to cloud platform (AWS, GCP, Azure)
- Add monitoring and logging
- Create frontend interface for predictions

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- Original Lab by Professor Ramin Mohammadi

---

