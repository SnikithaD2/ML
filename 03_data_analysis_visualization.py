# Question: Data Analysis and Visualization Using Dataset
#
# Description:
# Data analysis and visualization involve exploring datasets to extract insights and
# represent them graphically. Pandas is used for data manipulation, cleaning, and
# summarization, while Matplotlib helps create visual representations like scatter plots.
# Visualization makes patterns, trends, and relationships easier to understand. Combining
# analysis and visualization improves decision-making and data interpretation. This process
# is widely used in machine learning, business analytics, and scientific research for
# understanding complex datasets effectively and efficiently.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

dataset = load_iris()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

if hasattr(dataset, 'target'):
    df['target'] = dataset.target

print(df.head())
print(df.info())
print(df.describe())

if 'target' in df.columns:
    print(df['target'].value_counts())

x_feature = df.columns[0]
y_feature = df.columns[1]

plt.scatter(df[x_feature], df[y_feature],
            c=df['target'] if 'target' in df.columns else 'blue')

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title(f"{x_feature} vs {y_feature}")
plt.show()
