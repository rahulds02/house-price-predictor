from model import train_model
import os

def test_train_model():
    # Train the model
    model = train_model()

    # Test the model's predictions
    prediction = model.predict([[1000]])[0]
    assert prediction == 300000, f"Expected 300000 but got {prediction}"

    # Check if the model file exists
    assert os.path.exists("model.pkl"), "Model file not found!"