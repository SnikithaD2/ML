# Question: Simple Linear Regression
#
# Description:
# Simple linear regression is a supervised learning algorithm used to model the
# relationship between one independent variable and one dependent variable. It fits a
# straight line equation to observed data. The model calculates slope and intercept to
# predict values. It is widely used for forecasting and trend analysis. Linear regression
# assumes a linear relationship and helps understand how changes in input affect the output
# variable in predictive modeling tasks.

import pandas as pd
from sklearn.linear_model import LinearRegression

data = {'Experience': [1, 2, 3, 4, 5], 'Salary': [30000, 40000, 50500, 60000, 70000]}
df = pd.DataFrame(data)

X = df[['Experience']]
y = df['Salary']

model = LinearRegression()
model.fit(X, y)

predicted_salary = model.predict([[6]])

print(model.coef_)
print(model.intercept_)
print(predicted_salary[0])
