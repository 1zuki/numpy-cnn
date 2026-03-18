# numpy-cnn

A small convolutional neural network implemented mostly from scratch with **NumPy**.

This repository is a learning-focused project that builds the core pieces of a CNN manually instead of relying on a deep learning framework for the actual network logic. The training script uses `torchvision` to download EMNIST, but the model itself, forward pass, backward pass, and optimizer steps are implemented in NumPy.

## Highlights

- Manual **Conv2D** implementation using an `im2col` / `col2im` approach
- **MaxPool2D**, **ReLU**, **Flatten**, **Dense**, and **Dropout** layers
- **Cross-entropy loss**
- Adam-style parameter updates inside trainable layers
- Training on the **EMNIST digits** dataset
- Simple image-based digit inference with Pillow

## Repository structure

```text
numpy-cnn/
├── train.py         # Core CNN layers, model definition, dataset loading, and training loop
├── model.py         # Loads pickled weights and predicts from an image
├── test.py          # Small prediction test script
├── test-assets/     # Example digit images
└── LICENSE
```

## Current model architecture

The model defined in `train.py` is:

```text
Input (1 x 28 x 28)
→ Conv2D(1, 16, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2D(pool_size=2, stride=2)
→ Conv2D(16, 32, kernel_size=3, padding=1)
→ ReLU
→ MaxPool2D(pool_size=2, stride=2)
→ Dropout(0.25)
→ Flatten
→ Dense(32 * 7 * 7, 128)
→ ReLU
→ Dropout(0.5)
→ Dense(128, 47)
```

## Requirements

- Python 3.9+
- NumPy
- torch
- torchvision
- Pillow

Install dependencies with:

```bash
pip install numpy torch torchvision pillow
```

## Training

Run:

```bash
python train.py
```

What the training script currently does:

- downloads or loads the **EMNIST digits** dataset
- reshapes images to `(N, 1, 28, 28)`
- normalizes pixels to `[0, 1]`
- trains on the first 1000 samples only
- runs for 2 epochs with batch size 64
- saves weights with `pickle`

## Inference

To run a single prediction from an image file:

```bash
python model.py
```

The inference flow is:

1. Open an image with Pillow
2. Convert it to grayscale
3. Resize it to `28x28`
4. Invert the colors
5. Normalize to `[0, 1]`
6. Run a forward pass through the saved layers
7. Return the predicted class

## What this project is good for

This repo is best treated as an **educational CNN implementation**.

It is useful if you want to:

- understand how convolution can be implemented with matrix operations
- learn how manual backpropagation works layer by layer
- inspect a small NumPy-only deep learning pipeline
- experiment with building your own toy neural network framework

## Suggested next improvements

A good next pass on the repo would be:

- add a `requirements.txt`
- add command-line arguments for training and prediction
- include a short example workflow in the repo

## Example workflow

After cleaning up paths, a typical workflow would look like this:

```bash
python train.py
python model.py
python test.py
```

## License

- MIT
Learning project, use as you wish

## Special thanks

- @Acceleratorer - helping with the original idea and guiding
- @SamsonZhangTheSalmon on youtube - making the video on nn from scratch
- My innerself for not giving up

## Accel ni naritai