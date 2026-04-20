# Question: Central Tendency and Dispersion Measures
#
# Description:
# Central tendency and dispersion are statistical measures used to summarize datasets.
# Central tendency includes mean, median, and mode, representing the typical value.
# Dispersion measures such as variance and standard deviation indicate how spread out
# the data is. These metrics help in understanding data distribution, variability, and
# consistency. They are widely used in data analysis, machine learning, and decision-making
# to interpret numerical datasets effectively and draw meaningful conclusions from data.

import statistics as stats

arr= list(map(int,input("Enter the numbers:").split()))
n = len(arr)
mean = stats.mean(arr)
median = stats.median(arr)
mode = stats.multimode(arr)
variance = stats.variance(arr)
standard_deviation = stats.stdev(arr)
print("Mean =", mean)
print("Median =", median)
print("Mode =", mode)
print("Variance =", variance) 
print("Standard Deviation =", standard_deviation)
