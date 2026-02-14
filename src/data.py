import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def load_data():
    """
    Load the Breast Cancer dataset and return the features and target values.
    
    Returns:
        X (numpy.ndarray): The features (30 features).
        y (numpy.ndarray): The target values (0=malignant, 1=benign).
        feature_names (list): Names of the 30 features.
    """
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    feature_names = cancer.feature_names
    return X, y, feature_names

def split_data(X, y):
    """
    Split the data into training and testing sets.
    
    Args:
        X (numpy.ndarray): The features.
        y (numpy.ndarray): The target values.
    
    Returns:
        X_train, X_test, y_train, y_test (tuple): The split dataset.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test