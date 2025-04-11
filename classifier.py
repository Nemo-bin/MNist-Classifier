####################################
# Fully Connected MNIST classifier #
####################################

from tensorflow.keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize and reshape the data
x_train = x_train.reshape(-1, 784) / 255.0
x_test = x_test.reshape(-1, 784) / 255.0

class Classifier:
    def __init__(self, lr, hn1, hn2, alpha):
        self.lr = lr # Learning rate
        self.alpha = alpha # Momentum scaling parameter

        # Hidden Layer 1
        self.W1 = np.random.randn(784, hn1) * np.sqrt(2 / 784)
        self.b1 = np.zeros((1, hn1))
        self.M1_w = np.zeros((784, hn1))
        self.M1_b = np.zeros((1, hn1))

        # Hidden Layer 2
        self.W2 = np.random.randn(hn1, hn2) * np.sqrt(2 / hn1)
        self.b2 = np.zeros((1, hn2))
        self.M2_w = np.zeros((hn1, hn2))
        self.M2_b = np.zeros((1, hn2))

        # Output Layer
        lg, ug = -(1 / np.sqrt(hn2)), (1 / np.sqrt(hn2))
        self.W3 = lg + np.random.randn(hn2, 10) * (ug - lg)
        self.b3 = np.zeros((1, 10))
        self.M3_w = np.zeros((hn2, 10))
        self.M3_b = np.zeros((1, 10))

    def forward(self, X):
        # Hidden Layer 1
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.ReLU(self.Z1)

        # Hidden Layer 2
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.ReLU(self.Z2)

        # Output layer
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)

        return self.A3

    def backward(self, X, Y):
        m = X.shape[0]          
        Y = self.oneHot(Y.flatten())

        # Output layer gradients
        dZ3 = self.A3 - Y
        dW3 = (1 / m) * np.dot(self.A2.T, dZ3)
        db3 = (1 / m) * np.sum(dZ3, axis=0, keepdims=True)

        # Hidden layer 2
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.derivativeReLU(self.Z2)
        dW2 = (1 / m) * np.dot(self.A1.T, dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # Hidden layer 1
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.derivativeReLU(self.Z1)
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        self.M3_w = self.alpha * self.M3_w + self.lr * dW3
        self.M3_b = self.alpha * self.M3_b + self.lr * db3
        self.M2_w = self.alpha * self.M2_w + self.lr * dW2
        self.M2_b = self.alpha * self.M2_b + self.lr * db2
        self.M1_w = self.alpha * self.M1_w + self.lr * dW1
        self.M1_b = self.alpha * self.M1_b + self.lr * db1

        # Update parameters
        self.W3 -= self.M3_w
        self.b3 -= self.M3_b
        self.W2 -= self.M2_w
        self.b2 -= self.M2_b
        self.W1 -= self.M1_w
        self.b1 -= self.M1_b

    def train(self, X, Y, epochs, batch_size=128):
        for e in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                Y_batch = Y[i:i+batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, Y_batch)
            
            # Print training progress
            if e % 1 == 0:
                train_acc = self.accuracy(X, Y)
                print(f"Epoch {e}, Train Accuracy: {train_acc:.4f}")

    def predict(self, X):
        A3 = self.forward(X)
        return np.argmax(A3, axis=1)
    
    def accuracy(self, X, Y):
        predictions = self.predict(X)
        return np.mean(predictions == Y)

    # Helper Functions
    def oneHot(self, Y):
        one_hot = np.zeros((Y.size, 10))
        one_hot[np.arange(Y.size), Y] = 1
        return one_hot
    
    def ReLU(self, X):
        return np.maximum(0, X)

    def derivativeReLU(self, X):
        return (X > 0).astype(float)

    def softmax(self, X):
        e = np.exp(X - np.max(X, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    
# Initialize and train the classifier
classifier = Classifier(
    lr=0.01, 
    hn1=256,
    hn2=128,
    alpha=0.5
    )
classifier.train(x_train, y_train, epochs=50)

# Evaluate
train_accuracy = classifier.accuracy(x_train, y_train)
test_accuracy = classifier.accuracy(x_test, y_test)

print(f"\nFinal Train accuracy: {train_accuracy:.4f}")
print(f"Final Test accuracy: {test_accuracy:.4f}")