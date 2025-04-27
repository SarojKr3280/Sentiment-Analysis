from src.data_loader import load_features_labels, split_data
from src.vectorizer import load_vectorizer
from src.model_trainer import train_model, evaluate_model
import joblib
import json
import os

# 1. Load data
print("🔹 Loading data...")
X_sparse = joblib.load('data/processed/x_sparse_data.pkl')
X = X_sparse.toarray()

y = joblib.load('data/processed/y_data.pkl')

# 2. Load saved vectorizer (if needed later for predictions)
vectorizer = load_vectorizer('data/processed/tf_idf_vectorizer.pkl')

# 3. Split into train-test
print("🔹 Splitting data...")
X_train, X_test, y_train, y_test = split_data(X,y)


# 4. Load your trained model
print("🔹 Loading trained model...")
model = joblib.load('models/Lr_model.pkl')

# 5. Evaluate model
print("🔹 Evaluating model...")
metrics = evaluate_model(model, X_test, y_test)

# 6. Save metrics
print("🔹 Saving evaluation metrics...")
os.makedirs('report/metrics', exist_ok=True)

with open('reports/metrics/model_scores.json', 'w') as f:
    json.dump(metrics, f, indent=4)

print("✅ Pipeline completed successfully.")


