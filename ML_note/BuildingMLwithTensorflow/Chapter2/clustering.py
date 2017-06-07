'''
In this process, we will also gain two new tools: 
    the ability to generate synthetic sample sets from a collection of representative data structures
    via the scikit-learn library,and the ability to graphically plot our data and model results
k-means and knn
'''


'''
Learning from data-- unsupervised learning
Clustering
k-means: k-means tries to divide a set of samples in k disjoint groups or clusters
    using the mean value of the members as the main indicator. This point is normally
    called a Centroid. k-means is a naive method because it works by looking for
    the appropriate centroids but doesn't know a priori what the number of clusters is

k-nearest neighbors:
    a simple,classical method for clustering 
'''


'''
Practical examples for useful libraries
matplotlib  sklearn(dataset  Blob,circle, moon)

'''

# Sample synthetic data plotting
# import tensorflow as tf 
# import numpy as np 
# import matplotlib.pyplot as plt 

# with tf.Session() as sess:
#     fig, ax = plt.subplots()
#     ax.plot(sess.run(tf.random_normal([100])), 
#         sess.run(tf.random_normal([100])), 'o')
#     ax.set_title('Sample random plot for Tensorflow')
#     plt.show()
#     plt.savefig('result.png')



'''
Project 1  --k-means clustering on synthetic datasets
    we'll be using generated datasets that are specially crafted to have special
    properties. Two of the target properties are the possibility of linear separation
    of classes and the existence of clearly separated clusters(or not)
'''

import sklearn
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs, make_circles
import numpy as np 

# #1. generating the dataset
# centers = [(-2,-2), (-2,1.5), (1.5,-2), (2,1.5)]
# data, features = make_blobs(n_samples=200, centers=centers, n_features=2,
#                             cluster_std=0.8, shuffle=False, random_state=42)
# # print(features)
# # plt.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1],
# #             marker='o', s=250)
# # plt.show()


# # 2.Model architecture
# points = tf.Variable(data)
# centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K, 2]))
# cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))
# # draw the position of these centroids using matplotlib
# fig, ax = plt.subplots()
# ax.scatter(np.asarray(centers).transpose()[0], np.asarray(centers).transpose()[1],
#             marker='o', s=250)
# plt.show()


# #3. Loss function description and optimizer loop
# # N copies of all centroids, K copies of each point, and NxK copies of every point
# # so we can calculate the distances between each point and every centroid,for each dim
# rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
# rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
# sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids), reduction_indices=2)
# best_centroids = tf.argmin(sum_squares, 1)
# # Centroids will also be updated with a bucket: mean function


# #4.Stop condition
# # the new centroids and assignments don't change
# did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids, cluster_assignments))
# # use control_dependencies to calculate whether we need to update the centroids
# with tf.control_dependencies([did_assignments_change]):
#     do_updates = tf.group(
#                     centroids.assign(means),
#                     cluster_assignments.assign(best_centroids))




'''
Project2 -nearest neighbor on synthetic datasets
'''
# adding a bit more noise 
data, features = make_circles(n_samples=N, shuffle=True, noise=0.12, factor=0.4)

# model architecture
tr_data, tr_features = data[:cut], features[:cut]
te_data, te_features = data[cut:], features[cut:]
test = []

# loss function
distances = tf.reduce_sum(tf.square(tf.sub(i, tr_data)), reduction_indices=1)
neighbor = tf.arg_min(distances, 0)

# stop condition(we will finish once all the elements of the test partition have been visited)

