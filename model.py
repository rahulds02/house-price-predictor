import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train_model(data_path="data.csv", model_path="model.pkl"):
    # Load dataset
    data = pd.read_csv(data_path)
    X = data[["square_feet"]]
    y = data["price"]

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Save the model
    joblib.dump(model, model_path)
    print("Model trained and saved to", model_path)

    return model

# Ensure the file ends with exactly one newline
