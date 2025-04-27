from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def train_model(model, X_train, y_train):
    """Train the given model"""
    model.fit(X_train, y_train)
    print("✅ Model Training Successful. ")
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the given model and return a detailed dictionary"""
    y_pred = model.predict(X_test)

    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary'),
        'recall': recall_score(y_test, y_pred, average='binary'),
        'f1_score': f1_score(y_test, y_pred, average='binary'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }

    print("✅ Model Evaluation Successful. ")

    return results
