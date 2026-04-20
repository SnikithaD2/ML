# Question: Central Tendency and Dispersion Measures
#
# Description:
# Central tendency and dispersion are statistical measures used to summarize datasets.
# Central tendency includes mean, median, and mode, representing the typical value.
# Dispersion measures such as variance and standard deviation indicate how spread out
# the data is. These metrics help in understanding data distribution, variability, and
# consistency. They are widely used in data analysis, machine learning, and decision-making
# to interpret numerical datasets effectively and draw meaningful conclusions from data.

import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)

mean_val = df.mean()
median_val = df.median()
mode_val = df.mode().iloc[0]
std_dev = df.std()
variance = df.var()

stats_df = pd.DataFrame({
    'Mean': mean_val,
    'Median': median_val,
    'Mode': mode_val,
    'Std Dev': std_dev,
    'Variance': variance
})

print(stats_df)
