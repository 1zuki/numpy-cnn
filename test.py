from model import DigitRecognizer

def test_model():
    recognizer = DigitRecognizer("./numpy-cnn/cnn-models.pkl")

    for i in range(1, 10):
        cell_value = recognizer(f"./numpy-cnn/test-assets/{i}.png")
        print("Detected:", cell_value)
    
if __name__ == "__main__":
    test_model()