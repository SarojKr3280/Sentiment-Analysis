
import joblib

def load_features_labels(X_path, y_path):
    """Loads pre-saved features and labels."""
    X = joblib.load(X_path)
    y = joblib.load(y_path)
    print("✅ Data Loading Successful. ")
    return X, y


def split_data(X, y, test_size = 0.2, random_state = 42):
    """Splits the dataset into training and testing sets."""
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("✅ Data Splition Successful. ")
    return X_train, X_test, y_train, y_test