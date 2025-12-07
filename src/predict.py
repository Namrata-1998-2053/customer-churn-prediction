import joblib
import numpy as np
from preprocess import load_and_preprocess_data

# Predict churn for a new customer
def predict_new_customer(data):
    model = joblib.load("model/churn_model.pkl")

    # Data must match feature order used during training
    arr = np.array(data).reshape(1, -1)

    prediction = model.predict(arr)[0]

    return "Churn" if prediction == 1 else "Not Churn"

# Example usage
if __name__ == "__main__":
    sample_customer = [1, 35, 4, 50000, 1, 1, 1, 60000]  # gender, age, tenure, ...
    print(predict_new_customer(sample_customer))
