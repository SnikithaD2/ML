# Question: Multilayer Perceptron using AND
#
# Description:
# A Multilayer Perceptron (MLP) is a fully connected neural network with one or more
# hidden layers between the input and output layers. Unlike a single layer perceptron,
# MLP learns weights automatically during training using backpropagation. It uses
# scikit-learn's MLPClassifier to model the AND gate, where the output is 1 only when
# both inputs are 1. The hidden layer allows the model to learn internal representations,
# making it suitable for both linearly and non-linearly separable classification problems.

from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 0, 0, 1]

model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
model.fit(X, y)

print(model.predict(X))
