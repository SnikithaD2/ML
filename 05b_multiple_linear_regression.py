# Question: Multiple Linear Regression
#
# Description:
# Multiple linear regression extends simple regression by using multiple independent
# variables to predict a dependent variable. It models the relationship between several
# inputs and output using a linear equation. This approach improves prediction accuracy
# when multiple factors influence the outcome. It is commonly used in real-world applications
# like house price prediction. The model estimates coefficients for each feature, showing
# their contribution to the prediction while assuming a linear relationship.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_prices.csv")

df = df.fillna(df.median(numeric_only=True))

X = df[['area', 'bedrooms', 'age']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

sample_prediction = model.predict([[2500, 3, 10]])
print(sample_prediction[0])
print(model.score(X, y))

y_pred = model.predict(X)

plt.scatter(y, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression")
plt.show()
