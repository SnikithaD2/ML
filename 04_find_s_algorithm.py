# Question: FIND-S Algorithm
#
# Description:
# FIND-S is a concept learning algorithm used in machine learning to determine the most
# specific hypothesis that fits all positive training examples. It ignores negative examples
# and updates the hypothesis by generalizing attributes when necessary. Starting with the
# most specific hypothesis, it gradually adapts based on training data. FIND-S is simple
# and useful for understanding hypothesis space but has limitations, such as sensitivity
# to noise and inability to handle inconsistent data effectively.

data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

h = ['0', '0', '0', '0', '0', '0']

k = 0
for row in data:
    if row[-1] == 'Yes':
        for i in range(6):
            if h[i] == '0':
                h[i] = row[i]
            elif h[i] != row[i]:
                h[i] = '?'
    print("iteration:", k, h)
    k += 1

print("Final Hypothesis:", h)
