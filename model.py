import numpy as np
import pickle
from PIL import Image, ImageOps
import io
import train

class RedirectUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            module = 'train'
        return super().find_class(module, name)

class DigitRecognizer:
    def __init__(self, model_path="./numpy-cnn/cnn-models.pkl"):
        """Loads the trained NumPy CNN model from disk."""
        try:
            with open(model_path, 'rb') as f:
                self.layers = RedirectUnpickler(f).load()
            print(f"Model successfully loaded from {model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find {model_path}. Is the path correct bero?")

    def __call__(self, image_path):
        """Processes an image and returns the predicted digit."""
        img = Image.open(image_path).convert('L')
        img = img.resize((28, 28))
        img = ImageOps.invert(img) 
        img_arr = np.array(img, dtype=np.float32) / 255.0
        X = img_arr.reshape(1, 1, 28, 28)
        
        # run the Forward Pass
        for layer in self.layers:
            X = layer.forward(X)
        
        print(X)

        # geet the prediction
        prediction = np.argmax(X, axis=1)[0]
        
        return int(prediction)

if __name__ == "__main__":
    # instantiate the model
    recognizer = DigitRecognizer("./numpy-cnn/cnn-models.pkl")
    
    digit = recognizer("./numpy-cnn/test-assets/test.png")
    print(f"Predicted Digit: {digit}")