from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from data import load_data, split_data

def fit_model(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest Classifier and save the model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Testing features
        y_test: Testing labels
    """
    # Train Random Forest
    rf_classifier = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42
    )
    rf_classifier.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"MODEL TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Malignant', 'Benign']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save model
    joblib.dump(rf_classifier, "../model/breast_cancer_model.pkl")
    print(f"\n✅ Model saved to: ../model/breast_cancer_model.pkl")

if __name__ == "__main__":
    print("Loading data...")
    X, y, feature_names = load_data()
    print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    
    print("Training model...")
    fit_model(X_train, y_train, X_test, y_test)