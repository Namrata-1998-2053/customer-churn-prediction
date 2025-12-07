import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocess import load_and_preprocess_data

def train_model():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    print("Accuracy:", accuracy)
    print("ROC-AUC:", roc_auc)

    # Save model
    joblib.dump(model, "model/churn_model.pkl")
    print("Model saved to model/churn_model.pkl")

if __name__ == "__main__":
    train_model()
