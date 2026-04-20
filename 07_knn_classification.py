# Question: KNN Classification using Iris Dataset
#
# Description:
# K-Nearest Neighbour is a supervised learning algorithm used for classification and
# regression. It classifies data points based on the majority class of their nearest
# neighbors. The value of K determines how many neighbors influence the prediction.
# It is simple and effective for small datasets. Distance metrics like Euclidean distance
# are used to find neighbors. KNN does not require training but can be computationally
# expensive during prediction for large datasets.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = load_iris()

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
