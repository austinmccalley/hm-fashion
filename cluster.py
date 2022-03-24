from time import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA


def run_kmeans(data, n_clusters):
  t0 = time()
  kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4)
  kmeans.fit(data)

  print("done in %0.3fs" % (time() - t0))
  return kmeans

def run_gmm(data, n_clusters):
  t0 = time()
  gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
  gmm.fit(data)
  print("done in %0.3fs" % (time() - t0))
  return gmm

def visualize_gmm(gmm, data):
  labels = gmm.predict(data)
  plt.scatter(data[:, 0], data[:, 1], c=labels, s=40, cmap='viridis')
  plt.show()

def run_cluster(data, n_clusters=10, n_components=2, method='kmeans', visualize=False):
  reduced_data = PCA(n_components=n_components).fit_transform(data)

  if method == 'kmeans':
    model = run_kmeans(reduced_data, n_clusters)
  elif method == "gmm":
    model = run_gmm(reduced_data, n_clusters) 
    if visualize:
      visualize_gmm(model, reduced_data)
      return model

  if not visualize:
    return model

  # Step size of the mesh. Decrease to increase the quality of the VQ.
  h = .2     # point in the mesh [x_min, x_max]x[y_min, y_max].

  # Plot the decision boundary. For that, we will assign a color to each
  x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
  y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

  # Obtain labels for each point in mesh. Use last trained model.
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure(1)
  plt.clf()
  plt.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired, aspect='auto', origin='lower')

  plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
  # Plot the centroids as a white X
  centroids = model.cluster_centers_
  plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)

  plt.title('K-means clustering (PCA-reduced data)\n')
  plt.xlim(x_min, x_max)
  plt.ylim(y_min, y_max)
  plt.xticks(())
  plt.yticks(())
  plt.show()

  return model
