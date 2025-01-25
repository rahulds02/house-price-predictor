import unittest
from model import train_model


class TestModel(unittest.TestCase):
    def test_train_model(self):
        # Test training with mock data
        model = train_model(
            data_path="data.csv",
            model_path="test_model.pkl"
        )
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()

# Ensure the file ends with exactly one newline

