# Question: Multilayer Perceptron using OR
#
# Description:
# A Multilayer Perceptron (MLP) is a fully connected neural network with one or more
# hidden layers between the input and output layers. Unlike a single layer perceptron,
# MLP learns weights automatically during training using backpropagation. It uses
# scikit-learn's MLPClassifier to model the OR gate, where the output is 1 when at
# least one input is 1. The hidden layer enhances the model's learning capacity, enabling
# it to train effectively on simple logic gates and generalize to more complex classification
# tasks with minimal configuration changes.

from sklearn.neural_network import MLPClassifier

X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 1]

model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
model.fit(X, y)

print(model.predict(X))
