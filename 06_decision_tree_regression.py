# Question: Predict Humidity using Decision Tree Algorithm
#
# Description:
# Decision tree regression is a supervised learning technique used to predict continuous
# values by splitting data into branches based on feature conditions. It creates a tree-like
# structure where each node represents a decision rule. The model learns patterns from
# training data and predicts outcomes. It is easy to interpret and handles nonlinear
# relationships well. However, it may overfit if not properly controlled using parameters
# like depth and pruning techniques.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("daily_weather.csv")

X = df.drop('relative_humidity_9am', axis=1)
y = df['relative_humidity_9am']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred[:5])
print(r2_score(y_test, y_pred))
