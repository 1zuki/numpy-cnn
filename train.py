import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
import pickle
import time

# core engine: im2col & col2im
def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k.astype(int), i.astype(int), j.astype(int)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]

# nn layers
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # he initialization
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size**2))
        self.b = np.zeros((out_channels, 1))
        
        # adam optimizer states
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, X):
        self.X = X
        n, c, h, w = X.shape
        out_h = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (w + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.X_col = im2col_indices(X, self.kernel_size, self.kernel_size, self.padding, self.stride)
        W_col = self.W.reshape(self.out_channels, -1)

        out = W_col @ self.X_col + self.b
        return out.reshape(self.out_channels, out_h, out_w, n).transpose(3, 0, 1, 2)

    def backward(self, dout):
        dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        
        self.dW = (dout_reshaped @ self.X_col.T).reshape(self.W.shape)
        self.db = np.sum(dout_reshaped, axis=1, keepdims=True)

        W_reshape = self.W.reshape(self.out_channels, -1)
        dX_col = W_reshape.T @ dout_reshaped
        return col2im_indices(dX_col, self.X.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.W -= lr * (self.mW / (1 - beta1 ** self.t)) / (np.sqrt(self.vW / (1 - beta2 ** self.t)) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
        self.b -= lr * (self.mb / (1 - beta1 ** self.t)) / (np.sqrt(self.vb / (1 - beta2 ** self.t)) + eps)

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, X):
        self.X = X
        n, c, h, w = X.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1

        X_reshaped = X.reshape(n * c, 1, h, w)
        self.X_col = im2col_indices(X_reshaped, self.pool_size, self.pool_size, padding=0, stride=self.stride)

        self.max_indices = np.argmax(self.X_col, axis=0)
        out = self.X_col[self.max_indices, np.arange(self.X_col.shape[1])]
        return out.reshape(out_h, out_w, n, c).transpose(2, 3, 0, 1)

    def backward(self, dout):
        n, c, h, w = self.X.shape
        dX_col = np.zeros_like(self.X_col)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[self.max_indices, np.arange(self.X_col.shape[1])] = dout_flat

        dX = col2im_indices(dX_col, (n * c, 1, h, w), self.pool_size, self.pool_size, padding=0, stride=self.stride)
        return dX.reshape(self.X.shape)

class Dense:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros(out_features)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.t = 0

    def forward(self, X):
        self.X = X
        return X @ self.W + self.b

    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def step(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.t += 1
        self.mW = beta1 * self.mW + (1 - beta1) * self.dW
        self.vW = beta2 * self.vW + (1 - beta2) * (self.dW ** 2)
        self.W -= lr * (self.mW / (1 - beta1 ** self.t)) / (np.sqrt(self.vW / (1 - beta2 ** self.t)) + eps)

        self.mb = beta1 * self.mb + (1 - beta1) * self.db
        self.vb = beta2 * self.vb + (1 - beta2) * (self.db ** 2)
        self.b -= lr * (self.mb / (1 - beta1 ** self.t)) / (np.sqrt(self.vb / (1 - beta2 ** self.t)) + eps)

class Dropout:
    def __init__(self, rate=0.5):
        """
        rate: The probability of dropping a neuron (e.g., 0.5 means drop 50%).
        """
        self.rate = rate
        self.mask = None
        self.training = True # flag to toggle between train/test modes

    def forward(self, X):
        if self.training:
            # create a binary mask using a binomial distribution
            # scale by 1 / (1 - rate) to keep the expected value of the activations consistent
            self.mask = np.random.binomial(1, 1 - self.rate, size=X.shape) / (1.0 - self.rate)
            return X * self.mask
        else:
            # during inference -> sleep
            return X

    def backward(self, dout):
        # gradients only flow through the neurons that were kept active
        return dout * self.mask

class Flatten:
    def forward(self, X):
        self.X_shape = X.shape
        return X.reshape(X.shape[0], -1)
    def backward(self, dout):
        return dout.reshape(self.X_shape)

class ReLU:
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    def backward(self, dout):
        return dout * (self.X > 0)

class CrossEntropyLoss:
    def forward(self, logits, y):
        m = y.shape[0]
        # shift logits for numerical stability
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        log_probs = -np.log(self.probs[np.arange(m), y] + 1e-15)
        return np.sum(log_probs) / m

    def backward(self, y):
        m = y.shape[0]
        dout = self.probs.copy()
        dout[np.arange(m), y] -= 1
        return dout / m

# network assembly n trainning
class CNNModel:
    def __init__(self):
        self.layers = [
            Conv2D(1, 16, kernel_size=3, padding=1), ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            
            Conv2D(16, 32, kernel_size=3, padding=1), ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Dropout(rate=0.25), # drop 25% of spatial features
            
            Flatten(),
            Dense(32 * 7 * 7, 128), ReLU(),
            Dropout(rate=0.5),  # drop 50% of dense features to prevent memorization
            Dense(128, 47)
        ]
        self.loss_fn = CrossEntropyLoss()

    def train(self):
        """Sets the model to training mode (enables Dropout)."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        """Sets the model to evaluation mode (disables Dropout)."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, y):
        dout = self.loss_fn.backward(y)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def step(self, lr=0.001):
        for layer in self.layers:
            if hasattr(layer, 'step'):
                layer.step(lr)

    def save(self, filepath="license-plate/weights-plate.pkl"):
        self.eval() 
        with open(filepath, 'wb') as f:
            import pickle
            pickle.dump(self.layers, f)
            print(f"Model saved to {filepath}")

def load_emnist():
    print("Downloading/Loading EMNIST digits dataset")
    train_ds = torchvision.datasets.EMNIST(root='./data', split='digits', train=True, download=True)

    """cant be bothered changing"""

#    transform = transforms.Compose([
#        transforms.Grayscale(),
#        transforms.Resize((28,28)),
#       transforms.ToTensor()
#    ])

#    train_ds = datasets.ImageFolder(
#        root="./data/synthetic_digits",
#        transform=transform
#    )

    # extract transpose normalize
    X = train_ds.data.numpy().transpose(0, 2, 1)
    X = X.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    y = train_ds.targets.numpy()
    
    return X[:1000], y[:1000]

def start_training():
    X_train, y_train = load_emnist()
    model = CNNModel()
    
    epochs = 2
    batch_size = 64
    lr = 0.001
    num_batches = len(X_train) // batch_size
    
    print("\nTraining")
    for epoch in range(epochs):
        first_eta = True

        indices = np.random.permutation(len(X_train))
        X_shuffled, y_shuffled = X_train[indices], y_train[indices]
        
        start_time = time.time()
        for i in range(num_batches):
            X_batch = X_shuffled[i * batch_size : (i+1) * batch_size]
            y_batch = y_shuffled[i * batch_size : (i+1) * batch_size]
            
            logits = model.forward(X_batch)
            loss = model.loss_fn.forward(logits, y_batch)
            
            model.backward(y_batch)
            model.step(lr)

            
            if i % 50 == 0:
                acc = np.mean(np.argmax(logits, axis=1) == y_batch)
                print(f"Epoch {epoch+1}/{epochs} | Batch {i}/{num_batches} | Loss: {loss:.4f} | Acc: {acc:.4f}")

            if first_eta and i != 0 and i % 50 == 0:
                print(f"ETA: {(time.time() - start_time) * num_batches} \n")
                first_eta = False
                
        lr *= 0.95
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds.")
        
    model.save()

if __name__ == "__main__":
    start_training()