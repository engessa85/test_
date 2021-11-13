from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np

#--------------------------------------------------------------------------------
# 1. Create the following 3 dimensional datasets Z of 7 points
# Z = [[1, 2, 5], [1, 4, 6], [1, 0, 1], [4, 2, 9], [4, 4, 1], [4, 0, 7], [1, 8,9]]
#--------------------------------------------------------------------------------

Z = np.array([[1, 2, 5], [1, 4, 6], [1, 0, 1], [4, 2, 9], [4, 4, 1], [4, 0, 7], [1, 8,9]])


#--------------------------------------------------------------------------------
# 2. Cluster the dataset into 4 groups using k-means clustering. Use print to show the result in run.
#--------------------------------------------------------------------------------

print("-----------------------------------------------------------------")
kmeans = KMeans(n_clusters=4, random_state=0,max_iter=300).fit(Z)
print("cluster labels are:")
print(kmeans.labels_)


#--------------------------------------------------------------------------------
# 3. Evaluate the results, assuming the true clusters for the Z are:
# true_lables=[3, 2, 1, 0, 1, 0, 2]
#--------------------------------------------------------------------------------

print("-----------------------------------------------------------------")
labels_true = [3, 2, 1, 0, 1, 0, 2]
Evaluate = metrics.adjusted_rand_score(labels_true, kmeans.labels_)
print("Evaluation: ")
print(Evaluate)
print("-----------------------------------------------------------------")
