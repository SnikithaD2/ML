# Question: Multilayer Perceptron using XOR
#
# Description:
# A Multilayer Perceptron (MLP) is a fully connected neural network with one or more
# hidden layers between the input and output layers. The XOR gate is a classic example
# of a non-linearly separable problem that a single layer perceptron cannot solve.
# MLP overcomes this limitation using hidden layers and backpropagation to learn complex
# decision boundaries. The XOR gate outputs 1 only when inputs differ. This demonstrates
# how MLPs can solve problems that are impossible for simple perceptrons, making them
# powerful for real-world classification tasks.

from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
model.fit(X, y)

print(model.predict(X))
