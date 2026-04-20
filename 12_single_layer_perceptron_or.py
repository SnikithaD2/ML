# Question: Single Layer Perceptron using OR
#
# Description:
# A Single Layer Perceptron is the simplest form of a neural network consisting of
# one input layer and one output layer with no hidden layers. It uses manually set
# weights and a bias to compute a weighted sum of inputs and applies a step activation
# function to produce binary output. The OR gate outputs 1 when at least one input is 1.
# This demonstrates how a perceptron can model linearly separable boolean logic functions
# by adjusting the bias threshold to correctly classify all OR gate input combinations.

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

w = np.array([1, 1])
b = -0.5

for i in X:
    output = np.dot(i, w) + b
    print(1 if output > 0 else 0)
