# Question: Predict Rainfall using Logistic Regression
#
# Description:
# Logistic regression is a supervised learning algorithm used for binary classification
# problems. It predicts probabilities using a logistic function and maps them to class
# labels. Unlike linear regression, it outputs values between zero and one. It is widely
# used for classification tasks such as weather prediction. The model estimates relationships
# between features and outcomes and is easy to implement, interpret, and efficient for
# linearly separable classification problems.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("weatherAUS.csv")
df = df.dropna()

df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

X = df.select_dtypes(include=['float64', 'int64'])
y = df['RainTomorrow']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Rainfall Prediction")
plt.show()
