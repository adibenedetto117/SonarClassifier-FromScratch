import math
import numpy as np

inputs = []
targets = []

with open('sonar.all-data', 'r') as file:
    for line in file:
        row = line.strip().split(',')

        if row[-1] == "R":
            targets.append(0)
        else:
            targets.append(1)

        row.pop()
        inputs.append(row)

x = np.array(inputs, dtype=float)
y = np.array(targets, dtype=int)

# Combine inputs and targets into one array
data = np.column_stack((x, y))

# Shuffle the data
np.random.seed(42)  # For reproducibility
np.random.shuffle(data)

# Split the data back into inputs and targets
shuffled_inputs = data[:, :-1]
shuffled_targets = data[:, -1]

# Now split into training and test sets
train_size = 150
inputs = shuffled_inputs[:train_size]
targets = shuffled_targets[:train_size]
targets = targets.reshape(-1, 1)  # Reshape targets to be 2D as expected

test_inputs = shuffled_inputs[train_size:]
test_outputs = shuffled_targets[train_size:]

num_features = len(inputs[0])

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def log_loss(act, target):
    epsilon = 1e-15  # A small number to prevent log(0)
    act = np.clip(act, epsilon, 1 - epsilon)
    return -target * np.log(act) - (1 - target) * np.log(1 - act)


def predict(inputs, weights_hidden, weights_output, b_hidden, b_output):
    hidden_layer_input = np.dot(inputs, weights_hidden) + b_hidden
    hidden_layer_output = np.vectorize(relu)(hidden_layer_input)
    return sigmoid(np.dot(hidden_layer_output, weights_output) + b_output)

# Initialize network parameters
input_size = num_features
hidden_size1 = 60  # Size of the first hidden layer
hidden_size2 = 30  # Size of the second hidden layer
output_size = 1

# Weights and biases initialization with proper scaling
np.random.seed(42)  # For reproducibility
weights_hidden1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2. / input_size)
weights_hidden2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2. / hidden_size1)
weights_output = np.random.randn(hidden_size2, output_size) * np.sqrt(2. / hidden_size2)
bias_hidden1 = np.zeros(hidden_size1)
bias_hidden2 = np.zeros(hidden_size2)
bias_output = np.zeros(output_size)

# Training the network
epochs = 1000
lr = 0.1

for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(inputs, weights_hidden1) + bias_hidden1
    a1 = relu(z1)
    z2 = np.dot(a1, weights_hidden2) + bias_hidden2
    a2 = relu(z2)
    z_output = np.dot(a2, weights_output) + bias_output
    y_pred = sigmoid(z_output)

    # Compute the loss
    loss = np.mean(log_loss(targets, y_pred))

    # Backpropagation
    # Output layer gradients
    error_output = (y_pred - targets)  # Now, this will be a 2D array with the same shape as targets
    grad_weights_output = np.dot(a2.T, error_output) / len(targets)
    grad_bias_output = np.sum(error_output, axis=0) / len(targets)

    # Hidden layer 2 gradients
    error_hidden2 = np.dot(error_output, weights_output.T) * relu_derivative(z2)
    grad_weights_hidden2 = np.dot(a1.T, error_hidden2) / len(targets)
    grad_bias_hidden2 = np.sum(error_hidden2, axis=0) / len(targets)

    # Hidden layer 1 gradients
    error_hidden1 = np.dot(error_hidden2, weights_hidden2.T) * relu_derivative(z1)
    grad_weights_hidden1 = np.dot(inputs.T, error_hidden1) / len(targets)
    grad_bias_hidden1 = np.sum(error_hidden1, axis=0) / len(targets)

    # Update weights and biases
    weights_output -= lr * grad_weights_output
    bias_output -= lr * grad_bias_output
    weights_hidden2 -= lr * grad_weights_hidden2
    bias_hidden2 -= lr * grad_bias_hidden2
    weights_hidden1 -= lr * grad_weights_hidden1
    bias_hidden1 -= lr * grad_bias_hidden1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the network
correct = 0
for inp, true_output in zip(test_inputs, test_outputs):
    # Forward pass for testing
    hidden_input = relu(np.dot(inp, weights_hidden1) + bias_hidden1)
    hidden_output = relu(np.dot(hidden_input, weights_hidden2) + bias_hidden2)
    final_output = sigmoid(np.dot(hidden_output, weights_output) + bias_output)

    # Making a decision on the prediction
    predicted_output = 1 if final_output >= 0.5 else 0
    correct += (predicted_output == true_output)

accuracy = correct / len(test_outputs)
print(f"Test Accuracy: {accuracy:.2f}")