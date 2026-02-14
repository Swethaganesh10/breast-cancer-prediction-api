from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, Field
from predict import predict_data
import os

app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for predicting breast cancer diagnosis - Modified by Swetha Ganesh",
    version="2.0.0"
)

class CancerData(BaseModel):
    """
    Pydantic model for breast cancer tumor measurements.
    Using the 10 mean features (most important).
    """
    mean_radius: float = Field(..., ge=0, description="Mean radius of tumor")
    mean_texture: float = Field(..., ge=0, description="Mean texture")
    mean_perimeter: float = Field(..., ge=0, description="Mean perimeter")
    mean_area: float = Field(..., ge=0, description="Mean area")
    mean_smoothness: float = Field(..., ge=0, le=1, description="Mean smoothness (0-1)")
    mean_compactness: float = Field(..., ge=0, le=1, description="Mean compactness (0-1)")
    mean_concavity: float = Field(..., ge=0, le=1, description="Mean concavity (0-1)")
    mean_concave_points: float = Field(..., ge=0, le=1, description="Mean concave points (0-1)")
    mean_symmetry: float = Field(..., ge=0, le=1, description="Mean symmetry (0-1)")
    mean_fractal_dimension: float = Field(..., ge=0, le=0.1, description="Mean fractal dimension")
    
    # SE features (standard error)
    se_radius: float = Field(default=0.0, ge=0)
    se_texture: float = Field(default=0.0, ge=0)
    se_perimeter: float = Field(default=0.0, ge=0)
    se_area: float = Field(default=0.0, ge=0)
    se_smoothness: float = Field(default=0.0, ge=0)
    se_compactness: float = Field(default=0.0, ge=0)
    se_concavity: float = Field(default=0.0, ge=0)
    se_concave_points: float = Field(default=0.0, ge=0)
    se_symmetry: float = Field(default=0.0, ge=0)
    se_fractal_dimension: float = Field(default=0.0, ge=0)
    
    # Worst features
    worst_radius: float = Field(default=0.0, ge=0)
    worst_texture: float = Field(default=0.0, ge=0)
    worst_perimeter: float = Field(default=0.0, ge=0)
    worst_area: float = Field(default=0.0, ge=0)
    worst_smoothness: float = Field(default=0.0, ge=0)
    worst_compactness: float = Field(default=0.0, ge=0)
    worst_concavity: float = Field(default=0.0, ge=0)
    worst_concave_points: float = Field(default=0.0, ge=0)
    worst_symmetry: float = Field(default=0.0, ge=0)
    worst_fractal_dimension: float = Field(default=0.0, ge=0)

class CancerResponse(BaseModel):
    prediction: int
    diagnosis: str
    confidence: float
    probabilities: dict

@app.get("/", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    model_exists = os.path.exists("../model/breast_cancer_model.pkl")
    return {
        "status": "healthy",
        "model_loaded": model_exists,
        "api_version": "2.0.0",
        "author": "Swetha Ganesh"
    }

@app.get("/model-info")
async def model_info():
    """Get information about the model"""
    return {
        "model_type": "Random Forest Classifier",
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "n_features": 30,
        "n_samples": 569,
        "classes": {
            "0": "Malignant (Cancerous)",
            "1": "Benign (Non-cancerous)"
        },
        "modifications": [
            "Changed from Iris to Breast Cancer dataset",
            "Upgraded from Decision Tree to Random Forest",
            "Added confidence scores and probabilities",
            "Enhanced input validation with Pydantic",
            "Added model info endpoint"
        ]
    }

@app.post("/predict", response_model=CancerResponse)
async def predict_cancer(cancer_features: CancerData):
    """
    Predict breast cancer diagnosis based on tumor measurements.
    
    Returns:
        - prediction: 0 (Malignant) or 1 (Benign)
        - diagnosis: Human-readable diagnosis
        - confidence: Prediction confidence (0-1)
        - probabilities: Probabilities for each class
    """
    try:
        # Convert to feature array (all 30 features in order)
        features = [[
            cancer_features.mean_radius,
            cancer_features.mean_texture,
            cancer_features.mean_perimeter,
            cancer_features.mean_area,
            cancer_features.mean_smoothness,
            cancer_features.mean_compactness,
            cancer_features.mean_concavity,
            cancer_features.mean_concave_points,
            cancer_features.mean_symmetry,
            cancer_features.mean_fractal_dimension,
            cancer_features.se_radius,
            cancer_features.se_texture,
            cancer_features.se_perimeter,
            cancer_features.se_area,
            cancer_features.se_smoothness,
            cancer_features.se_compactness,
            cancer_features.se_concavity,
            cancer_features.se_concave_points,
            cancer_features.se_symmetry,
            cancer_features.se_fractal_dimension,
            cancer_features.worst_radius,
            cancer_features.worst_texture,
            cancer_features.worst_perimeter,
            cancer_features.worst_area,
            cancer_features.worst_smoothness,
            cancer_features.worst_compactness,
            cancer_features.worst_concavity,
            cancer_features.worst_concave_points,
            cancer_features.worst_symmetry,
            cancer_features.worst_fractal_dimension
        ]]
        
        prediction, probabilities = predict_data(features)
        
        diagnosis = "Benign (Non-cancerous)" if prediction == 1 else "Malignant (Cancerous)"
        confidence = float(max(probabilities))
        
        return CancerResponse(
            prediction=int(prediction),
            diagnosis=diagnosis,
            confidence=round(confidence, 4),
            probabilities={
                "malignant": round(float(probabilities[0]), 4),
                "benign": round(float(probabilities[1]), 4)
            }
        )
    
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503, 
            detail="Model not found. Please train the model first by running train.py"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction error: {str(e)}"
        )