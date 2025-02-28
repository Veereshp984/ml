import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

# Load the dataset
dataset = pd.read_csv("Iris.csv")
X = dataset.iloc[:, 1:-1]  # Skip the first column (ID) and last column (Class)
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
y = [label[c] for c in dataset.iloc[:, -1]]

# Set up the colormap for visualization
colormap = np.array(['red', 'lime', 'black'])

# Plotting
plt.figure(figsize=(14, 7))

# Real Plot
plt.subplot(1, 3, 1)
plt.title('Real')
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[y])

# K-Means Clustering
model = KMeans(n_clusters=3, random_state=0).fit(X)
plt.subplot(1, 3, 2)
plt.title('KMeans')
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[model.labels_])
print('The accuracy score of K-Means: ', metrics.accuracy_score(y, model.labels_))
print('The Confusion matrix of K-Means:\n', metrics.confusion_matrix(y, model.labels_))

# Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=3, random_state=0).fit(X)
y_cluster_gmm = gmm.predict(X)
plt.subplot(1, 3, 3)
plt.title('GMM Classification')
plt.scatter(X.PetalLengthCm, X.PetalWidthCm, c=colormap[y_cluster_gmm])
print('The accuracy score of GMM: ', metrics.accuracy_score(y, y_cluster_gmm))
print('The Confusion matrix of GMM:\n', metrics.confusion_matrix(y, y_cluster_gmm))

plt.show()