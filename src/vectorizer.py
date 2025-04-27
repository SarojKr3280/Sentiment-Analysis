import joblib

def save_vectorizer(vectorizer, filepath):
    """Save the TF-IDF vectorizer to a file."""
    joblib.dump(vectorizer, filepath)
    # print(f"✅ Vectorizer saved successfully at: {filepath}")

def load_vectorizer(filepath):
    """Load the TF-IDF vectorizer from a file."""
    vectorizer = joblib.load(filepath)
    print(f"✅ Vectorizer loaded successfully")
    return vectorizer