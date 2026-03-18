from model import DigitRecognizer

def test_model():
    recognizer = DigitRecognizer("./numpy-cnn/cnn-models.pkl")

    for i in range(1, 10):
        cell_value = recognizer(f"./numpy-cnn/test-assets/{i}.png")
        print("Detected:", cell_value)

def expected_output():
    print("Expected output after training:\n1, 2, 3, 4, 5, 0, 7, 0, 9") # bad font
if __name__ == "__main__":
    test_model()