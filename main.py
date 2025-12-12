import numpy as np
from tensorflow.keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten images from 28x28 to 784-dimensional vectors
x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
x_test = x_test.reshape(-1, 784).astype("float32") / 255.0

y_train = y_train.astype(int)
y_test = y_test.astype(int)

def one_hot(y):
    onehot = np.zeros((y.size, 10))
    onehot[np.arange(y.size), y] = 1
    return onehot


y_hot = one_hot(y_train)
Y_test_hot = one_hot(y_test)

np.random.seed(0)

W1 = np.random.randn(784, 128) * 0.01
B1 = np.zeros((1, 128))

W2 = np.random.randn(128, 10) * 0.01
B2 = np.zeros((1, 10))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(x):
    Z1 = x.dot(W1) + B1
    A1 = relu(Z1)

    Z2 = A1.dot(W2) + B2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def loss(pred, true):
    return -np.mean(np.sum(true * np.log(pred + 1e-8), axis=1))

def backward(x, Z1, A1, Z2, A2, y_true, lr=0.01):
    global W1, B1, W2, B2

    m = x.shape[0]
    
    # Output layer gradients
    dZ2 = (A2 - y_true) / m
    dW2 = A1.T.dot(dZ2)
    dB2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradients
    dZ1 = dZ2.dot(W2.T) * relu_derivative(Z1)
    dW1 = x.T.dot(dZ1)
    dB1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update weights
    W1 -= lr * dW1
    B1 -= lr * dB1
    W2 -= lr * dW2
    B2 -= lr * dB2

epochs = 5
batch_size = 64

for epoch in range(epochs):
    idx = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[idx]
    y_hot_shuffled = y_hot[idx]

    for i in range(0, len(x_train_shuffled), batch_size):
        x_batch = x_train_shuffled[i:i+batch_size]
        y_batch = y_hot_shuffled[i:i+batch_size]

        z1, a1, z2, a2 = forward(x_batch)
        backward(x_batch, z1, a1, z2, a2, y_batch, lr=0.05)

    _, _, _, logits = forward(x_test)
    pred = np.argmax(logits, axis=1)
    acc = np.mean(pred == y_test)
    print(f"Epoch {epoch} - Accuracy: {acc:.4f}")

# Test on a sample
idx = 0
sample = x_test[idx:idx+1]
_, _, _, out = forward(sample)
print("Predicted:", np.argmax(out))
print("Actual Value:", y_test[idx])

# Save the model
model_data = {
    'W1': W1,
    'B1': B1,
    'W2': W2,
    'B2': B2
}
np.save('mnist_model.npy', model_data)
print("\nModel saved as 'mnist_model.npy'")