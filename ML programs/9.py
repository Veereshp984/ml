import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Kernel function for weights
def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))  # Corrected to np.eye
    for j in range(m):
        diff = point - xmat[j]  # Use xmat instead of undefined X
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k**2))
    return weights

# Function to calculate local weights
def localWeight(point, xmat, ymat, k):
    wei = kernel(point, xmat, k)
    W = (xmat.T * (wei * xmat)).I * (xmat.T * (wei * ymat.T))  # Matrix operations
    return W

# Function for local weighted regression
def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat)
    ypred = np.zeros(m)
    for i in range(m):
        ypred[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return ypred

# Load data points
data = pd.read_csv('10-dataset.csv')
bill = np.array(data['total_bill'])  # Corrected column name references
tip = np.array(data['tip'])

# Preparing and adding 1's to bill for bias term
mbill = np.mat(bill).T  # Transpose to align dimensions
mtip = np.mat(tip).T
m = np.shape(mbill)[0]
one = np.mat(np.ones(m)).T  # Corrected np1 to np
X = np.hstack((one, mbill))  # Horizontally stack ones and bills

# Set kernel bandwidth (k)
k = 0.5
ypred = localWeightRegression(X, mtip, k)
SortIndex = X[:, 1].argsort(0)  # Sort based on the second column (bill)
xsort = X[SortIndex][:, 0]  # Proper sorting for x values

# Plotting
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(bill, tip, color='green', label='Data Points')  # Scatter plot
ax.plot(X[SortIndex][:, 1], ypred[SortIndex], color='red', linewidth=2, label='Prediction')  # Prediction line
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Local Weighted Regression')
plt.legend()
plt.show()
