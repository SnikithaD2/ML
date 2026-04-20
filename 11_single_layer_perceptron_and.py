# Question: Single Layer Perceptron using AND
#
# Description:
# A Single Layer Perceptron is the simplest form of a neural network consisting of
# one input layer and one output layer with no hidden layers. It uses manually set
# weights and a bias to compute a weighted sum of inputs and applies a step activation
# function to produce binary output. The AND gate outputs 1 only when all inputs are 1.
# This demonstrates how a perceptron can model linearly separable boolean logic functions
# using fixed weights and threshold-based decision making.

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

w = np.array([1, 1])
b = -1.5

for i in X:
    output = np.dot(i, w) + b
    print(1 if output > 0 else 0)
