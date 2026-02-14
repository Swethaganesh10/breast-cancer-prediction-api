import joblib
import os

def predict_data(X):
    """
    Predict the class labels for input data.
    
    Args:
        X: Input features (30 features)
    
    Returns:
        prediction: Predicted class (0 or 1)
        probabilities: Prediction probabilities [prob_malignant, prob_benign]
    """
    model_path = "../model/breast_cancer_model.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
    
    model = joblib.load(model_path)
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    
    return prediction[0], probabilities[0]