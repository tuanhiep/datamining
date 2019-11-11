# Question 1. (Data Pre-processing)
# 1a. Remove all rows that contain missing values.
# 1b. Remove all outliers. To discard the outliers, we can compute the Z-score for each attribute and
# remove rows containing attributes with abnormally high or low Z-score (e.g., if Z ≥ 3 or Z ≤ -3).
# After performing the above tasks, save the new data to data.csv and submit it. Note that data.csv
# contains the average ratings as in the original data, not the Z-scores.
# Use the new data (data.csv) generated in Question 1, perform the following tasks.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

df = pd.read_csv('csv/google_review_ratings.csv')
print('Number of rows in original data = %d' % (df.shape[0]))
raw_data = df.dropna()
print('Number of rows after discarding missing values = %d' % (raw_data.shape[0]))

# 1b. Remove all outliers. To discard the outliers, we can compute the Z-score for each attribute and
# remove rows containing attributes with abnormally high or low Z-score (e.g., if Z ≥ 3 or Z ≤ -3).
# After performing the above tasks, save the new data to data.csv and submit it. Note that data.csv
# contains the average ratings as in the original data, not the Z-scores.
clean_data = raw_data.drop(['User'], axis=1)
z_score = (clean_data - clean_data.mean()) / clean_data.std()
print('Number of rows before discarding outliers = %d' % (z_score.shape[0]))
# Find all the rows that have all of its values Z>-3 and Z<=3
clean_z_score = z_score.loc[((z_score > -3).sum(axis=1) == 24) & ((z_score <= 3).sum(axis=1) == 24), :]
print('Number of rows after discarding missing values = %d' % (clean_z_score.shape[0]))
data_save = raw_data[raw_data.index.isin(clean_z_score.index)]

np.savetxt('csv/data.csv', data_save, delimiter=', ', fmt='%s')

# Question 2. (k-means)
# We want to train a k-means model.
# 2a. We will use the Silhouette coefficient to select the best number of clusters (k) from [1,20]. For
# each k, run k-means 10 times and compute the average Silhouette coefficient across 10 running
# times and clusters. Plot the average Silhouette coefficients for different k. Submit the plot. What is
# the best k?

from sklearn.metrics import silhouette_score

data = data_save.drop(['User'], axis=1)
numClusters = np.arange(2, 21)
silhouette_k_mean = []
for k in numClusters:
    k_means = cluster.KMeans(n_clusters=k, max_iter=10, random_state=1)
    predicts = k_means.fit_predict(data)
    score = silhouette_score(data, predicts, metric='euclidean')
    print("K-means: For k = {}, silhouette score is {}".format(k, score))
    silhouette_k_mean.append(score)

plt.plot(numClusters, silhouette_k_mean)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('K-means')
#plt.show()

# Best number of K resulting the highest silhouette average score
best_k = numClusters[silhouette_k_mean.index(max(silhouette_k_mean))]

print("Best K for number of clusters is {}".format(best_k))

# # 2b. Train a k-means model with the best k above. Report the centroids of clusters.
#
k_means = cluster.KMeans(n_clusters=best_k, max_iter=10, random_state=1)
k_means.fit(data)
labels = k_means.labels_
centroids = k_means.cluster_centers_
centroids_data_frame = pd.DataFrame(centroids, columns=data.columns)
print(centroids_data_frame)
centroids_data_frame.to_csv('csv/centroids.csv')
#
# # 2c. Project the data using PCA with two principal components. From the projected data, draw a scatter plot using
# # cluster IDs as labels (legends). Are the found clusters well separated in the visualization? Is that observation
# # consistent with the computed Silhouette coefficient?
#
#
numComponents = 2
pca = PCA(n_components=numComponents)
pca.fit(data)

projected = pca.transform(data)
projected = pd.DataFrame(projected, columns=['pc1', 'pc2'])
projected['label'] = labels
print(projected)
print(labels)
print(labels.shape)
plt.clf()
# Scatter plot
for label in labels:
    d = projected[projected['label'] == label]
    plt.scatter(d['pc1'], d['pc2'])
plt.title('K-means')

#plt.show()
#
# # Question 3. (k-means++)
# # Perform the tasks in Question 2 with k-means++.
#
# # We want to train a k-means++ model.
# # 3a. We will use the Silhouette coefficient to select the best number of clusters (k) from [1,20]. For
# # each k, run k-means 10 times and compute the average Silhouette coefficient across 10 running
# # times and clusters. Plot the average Silhouette coefficients for different k. Submit the plot. What is
# # the best k?
#
data = data_save.drop(['User'], axis=1)
numClusters = np.arange(2, 21)
silhouette_k_mean_plus = []
for k in numClusters:
    k_means = cluster.KMeans(n_clusters=k, init='k-means++', max_iter=10, random_state=1)
    predicts = k_means.fit_predict(data)
    score = silhouette_score(data, predicts, metric='euclidean')
    print("K-means++ : For k = {}, silhouette score is {}".format(k, score))
    silhouette_k_mean_plus.append(score)

plt.plot(numClusters, silhouette_k_mean_plus)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('K-means++')
#plt.show()
#
# # Best number of K resulting the highest silhouette average score
best_k = numClusters[silhouette_k_mean_plus.index(max(silhouette_k_mean_plus))]

print("Best K for number of clusters with K-means++ algorithm is {}".format(best_k))
#
# # 3b. Train a k-means model with the best k above. Report the centroids of clusters.
#
k_means = cluster.KMeans(n_clusters=best_k, init='k-means++', max_iter=10, random_state=1)
k_means.fit(data)
labels = k_means.labels_
centroids = k_means.cluster_centers_
centroids_data_frame = pd.DataFrame(centroids, columns=data.columns)
print(centroids_data_frame)
centroids_data_frame.to_csv('csv/centroids-k-means++.csv')
#
# # 3c. Project the data using PCA with two principal components. From the projected data, draw a scatter plot using
# # cluster IDs as labels (legends). Are the found clusters well separated in the visualization? Is that observation
# # consistent with the computed Silhouette coefficient?
#
#
numComponents = 2
pca = PCA(n_components=numComponents)
pca.fit(data)

projected = pca.transform(data)
projected = pd.DataFrame(projected, columns=['pc1', 'pc2'])
projected['label'] = labels
print(projected)
print(labels)
print(labels.shape)
plt.clf()
# # Scatter plot
for label in labels:
    d = projected[projected['label'] == label]
    plt.scatter(d['pc1'], d['pc2'])
plt.title('K-means++')

#plt.show()

# Question 4. (Agglomerative Clustering)
# Perform the tasks in Question 2 with Agglomerative Clustering. One thing to note is that for each k, running
# Agglomerative Clustering one time is good enough.


# 4a. We will use the Silhouette coefficient to select the best number of clusters (k) from [1,20]. For
# each k, run Agglomerative Clustering and compute the average Silhouette coefficient across 10 running
# times and clusters. Plot the average Silhouette coefficients for different k. Submit the plot. What is
# the best k?


data = data_save.drop(['User'], axis=1)
numClusters = np.arange(2, 21)
silhouette_agglomerative = []
for k in numClusters:
    model = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward')
    predicts = model.fit_predict(data)
    score = silhouette_score(data, predicts, metric='euclidean')
    print("Agglomerative clustering: For k = {}, silhouette score is {}".format(k, score))
    silhouette_agglomerative.append(score)

plt.plot(numClusters, silhouette_agglomerative)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Agglomerative Clustering')
#plt.show()

# Best number of K resulting the highest silhouette average score
best_k = numClusters[silhouette_agglomerative.index(max(silhouette_agglomerative))]

print("Best K for number of clusters with Agglomerative clustering is {}".format(best_k))

# 4b. Train an Agglomerative Clustering model with the best k above. This model don't need to compute the centroids of
# clusters but we can calculate them manually by mean function

model = AgglomerativeClustering(n_clusters=best_k, affinity='euclidean', linkage='ward')
model.fit(data)
labels = model.labels_

predicts = data.copy()
predicts['label'] = labels
centroids = []
for label in range(0, best_k):
    d = predicts[predicts['label'] == label]
    centroids.append(d.mean(0))
centroids = np.array(centroids)
centroids_data_frame = pd.DataFrame(centroids, columns=predicts.columns)
print(centroids_data_frame)
centroids_data_frame.to_csv('csv/centroids-agglomerative-clustering.csv')

# 4c. Project the data using PCA with two principal components. From the projected data, draw a scatter plot using
# cluster IDs as labels (legends). Are the found clusters well separated in the visualization? Is that observation
# consistent with the computed Silhouette coefficient?


numComponents = 2
pca = PCA(n_components=numComponents)
pca.fit(data)

projected = pca.transform(data)
projected = pd.DataFrame(projected, columns=['pc1', 'pc2'])
projected['label'] = labels
print(projected)
print(labels)
print(labels.shape)
plt.clf()
# Scatter plot
for label in labels:
    d = projected[projected['label'] == label]
    plt.scatter(d['pc1'], d['pc2'])
plt.title('Agglomerative Clustering')

#plt.show()

# Question 5. (Gaussian Mixture Model)
# Perform the tasks in Question 2 with Gaussian Mixture Model.

# 5a. We will use the Silhouette coefficient to select the best number of clusters (k) from [1,20]. For
# each k, run Gaussian Mixture Model Clustering and compute the average Silhouette coefficient across 10 running
# times and clusters. Plot the average Silhouette coefficients for different k. Submit the plot. What is
# the best k?
from sklearn.mixture import GaussianMixture

data = data_save.drop(['User'], axis=1)
numClusters = np.arange(2, 21)
silhouette_gaussian = []
for k in numClusters:
    gmm = GaussianMixture(n_components=k)
    predicts = gmm.fit_predict(data)
    score = silhouette_score(data, predicts, metric='euclidean')
    print("Gaussian Mixture Model clustering: For k = {}, silhouette score is {}".format(k, score))
    silhouette_gaussian.append(score)

plt.plot(numClusters, silhouette_gaussian)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Gaussian Mixture Model Clustering')
#plt.show()

# Best number of K resulting the highest silhouette average score
best_k = numClusters[silhouette_gaussian.index(max(silhouette_gaussian))]

print("Best K for number of clusters with Gaussian Mixture Model clustering is {}".format(best_k))

# 5b. Train an Gaussian Mixture Model Clustering model with the best k above.
gmm = GaussianMixture(n_components=best_k)
gmm.fit(data)
labels = gmm.predict(data)

predicts = data.copy()
predicts['label'] = labels
centroids = []
for label in range(0, best_k):
    d = predicts[predicts['label'] == label]
    centroids.append(d.mean(0))
centroids = np.array(centroids)
centroids_data_frame = pd.DataFrame(centroids, columns=predicts.columns)
print(centroids_data_frame)
centroids_data_frame.to_csv('csv/centroids-gaussian-mixture-model-clustering.csv')

# 5c. Project the data using PCA with two principal components. From the projected data, draw a scatter plot using
# cluster IDs as labels (legends). Are the found clusters well separated in the visualization? Is that observation
# consistent with the computed Silhouette coefficient?


numComponents = 2
pca = PCA(n_components=numComponents)
pca.fit(data)

projected = pca.transform(data)
projected = pd.DataFrame(projected, columns=['pc1', 'pc2'])
projected['label'] = labels
print(projected)
print(labels)
print(labels.shape)
plt.clf()
# Scatter plot
for label in labels:
    d = projected[projected['label'] == label]
    plt.scatter(d['pc1'], d['pc2'])
plt.title('Gaussian Mixture Model Clustering')

#plt.show()

# Question 6. (Comparison)
# Draw a plot that includes the average Silhouette coefficients across different k of all the methods
# above. Submit the plot. Which method is better in terms of the average Silhouette coefficient?

plt.clf()
plt.plot(numClusters, silhouette_k_mean)
plt.plot(numClusters, silhouette_k_mean_plus)
plt.plot(numClusters, silhouette_agglomerative)
plt.plot(numClusters, silhouette_gaussian)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Model Clustering')
plt.legend(['K-mean', 'K-mean++', 'Agglomerative', 'Gaussian Mixture Model'], loc='upper left')
plt.show()
