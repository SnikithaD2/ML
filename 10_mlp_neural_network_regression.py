# Question: MLP Neural Network for Regression
#
# Description:
# A Multi-Layer Perceptron is a type of artificial neural network used for regression
# and classification tasks. It consists of input, hidden, and output layers with
# interconnected neurons. Each neuron applies weights and activation functions to learn
# patterns from data. MLP models can capture complex nonlinear relationships. They are
# widely used in deep learning applications. Proper tuning of layers, neurons, and epochs
# is important for achieving accurate predictions.

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("housing.csv")

df = df.dropna()
df = df.select_dtypes(include=['int64', 'float64'])

X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=10)

predictions = model.predict(X_test)

print(predictions[:5])

mse = model.evaluate(X_test, y_test)
print(mse)
