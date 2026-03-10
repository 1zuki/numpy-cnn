from cnn.model import DigitRecognizer

def test_model():
    recognizer = DigitRecognizer("./sudoku/cnn/pretrained-weight.pkl")

    for i in range(1, 10):
        cell_value = recognizer(f"./sudoku/cnn/test-assets/{i}.png")
        print("Detected:", cell_value)

def expected_output():
    print("Expected output after training:\n1, 2, 3, 4, 5, 0, 7, 0, 9")

if __name__ == "__main__":
    test_model()