# Question: Arithmetic Array Operations Using Python Libraries
#
# Description:
# Arithmetic array operations involve performing mathematical computations on arrays
# using Python libraries like NumPy, math, statistics, and SciPy. NumPy enables efficient
# element-wise operations such as addition, subtraction, multiplication, and division.
# The math library handles scalar operations, while statistics provides measures like mean
# and variance. SciPy extends functionality with advanced operations like matrix inversion
# and determinants. These tools are essential for scientific computing, numerical analysis,
# and data processing tasks in Python.

import math
import statistics
import numpy as np
from scipy import linalg

arr1 = np.array([10, 20, 30, 40])
arr2 = np.array([2, 4, 5, 8])

print(np.add(arr1, arr2))
print(np.subtract(arr1, arr2))
print(np.multiply(arr1, arr2))
print(np.divide(arr1, arr2))
print(np.power(arr1, arr2))
print(np.mod(arr1, arr2))

a = 16
b = 3

print(math.sqrt(a))
print(math.pow(a, b))
print(math.fmod(a, b))

data = [10, 20, 30, 40, 50]

print(statistics.mean(data))
print(statistics.median(data))
print(statistics.variance(data))
print(statistics.stdev(data))

matrix = np.array([[1, 2], [3, 4]])

print(linalg.det(matrix))
print(linalg.inv(matrix))
