import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(path="data/churn_data.csv"):
    df = pd.read_csv(path)

    # Encode categorical features
    label_encoder = LabelEncoder()
    df["gender"] = label_encoder.fit_transform(df["gender"])  # Male=1, Female=0

    # Features and target
    X = df.drop(["customer_id", "churn"], axis=1)
    y = df["churn"]

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test
