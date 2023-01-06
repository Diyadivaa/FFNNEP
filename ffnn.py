import numpy as np
import pandas as pd

# Load the data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data', 
                 header=None, 
                 names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])

# Preprocess the data
df = df.replace('?', np.nan).dropna()
df['sex'] = df['sex'].map({0: 1, 1: -1})
df['thal'] = df['thal'].map({3: 1, 6: 0, 7: 0})
df['ca'] = df['ca'].map({0: 1, 1: 0, 2: 0, 3: 0})
df = df.astype(float)

# Split the data into training and test sets
X = df.drop('num', axis=1).values
y = df['num'].values
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Normalize the data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# Build the model
n_inputs = X_train.shape[1]
n_hidden = 10
n_outputs = 1
learning_rate = 0.01

# Initialize weights and biases
W1 = np.random.randn(n_inputs, n_hidden)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, n_outputs)
b2 = np.zeros(n_outputs)

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Train the model
losses = []
for i in range(1000):
    # Forward pass
    z1 = X_train.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    a2 = sigmoid(z2)
    
    # Compute the loss
    loss = np.square(a2 - y_train).sum()
    losses.append(loss)

    
    # Backpropagation
    dz2 = a2 - y_train
    dw2 = a1.T.dot(dz2)
    db2 = dz2.sum(axis=0)
    dz1 = dz2.dot(W2.T) * sigmoid_derivative(z1)
    dw1 = X_train.T.dot(dz1)
    db1 = dz1.sum(axis=0)
    
    # Update weights and biases
    W1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dw2
    b2 -= learning_rate * db2

# Plot the loss over time
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
