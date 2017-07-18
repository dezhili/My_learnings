import matplotlib.pyplot as plt
import numpy as np


# # 第一步：导入数据集
# from sklearn.datasets import load_iris
# iris = load_iris()
# X, y = iris.data, iris.target
# print(X.shape)
# print(y.shape)
#
# # 第二步：划分数据集
# from sklearn.model_selection import train_test_split
# train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.75,
#                                                     random_state=123, stratify=y)
# print('All:', np.bincount(y)/float(len(y)) * 100.0) #[ 33.33333333  33.33333333  33.33333333]
# print('Training:', np.bincount(train_y)/float(len(train_y)) * 100.0)#[ 33.92857143  33.03571429  33.03571429]
# print('Test:', np.bincount(test_y)/float(len(test_y)) * 100.0)#[ 31.57894737  34.21052632  34.21052632]
#
# from sklearn.neighbors import KNeighborsClassifier
# classifier = KNeighborsClassifier()
# print(classifier)
#
# # 第四步：训练分类器
# classifier.fit(train_X, train_y)
#
# # 第五步：使用分类器预测测试数据
# pred_y = classifier.predict(test_X)
# #[2 0 1 2 0 0 1 2 1 0 1 0 2 2 1 2 0 0 0 0 0 0 1 2 0 2 2 2 2 1 1 2 1 1 2 1 2 1]
# print(pred_y)
# print(test_y)
#
# # 第六步：计算测试数据上的预测正确率
# print('Fraction Correct [Accuracy]:')
# accuracy = np.sum(pred_y == test_y) / float(len(test_y))
# print(accuracy)
#
# print('Samples correctly classified:')
# correct_idx = np.where(pred_y == test_y)[0]
# #[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 26 27 28 29 30 31 32 33 34 35 36 37]
# print(correct_idx)
#
# print('\nSamples incorrectly classified')
# incorrect_idx = np.where(pred_y != test_y)[0]
# print(incorrect_idx)
#
#
# # 第七步：可视化结果
# colors = ['darkblue', 'darkgreen', 'gray']
#
# for n, color in enumerate(colors):
#     idx = np.where(test_y == n)[0]
#     plt.scatter(test_X[idx, 1], test_X[idx, 2], color=color, label='Class %s'%n)
#
# plt.scatter(test_X[incorrect_idx, 1], test_X[incorrect_idx, 2], color='darkred')
#
# plt.xlabel('sepal width [cm]')
# plt.ylabel('petal length [cm]')
# plt.legend(loc=3)
# plt.title('Iris Classification results')
# plt.show()


print('----------------------------------第二个例子------------------------------------')
#-------------------------------------   第二个例子--------------------------------------

from sklearn.datasets import make_blobs

X, y = make_blobs(centers=2, random_state=0)
print('X ~ n_samples x n_features:', X.shape)
print('y ~ n_samples:', y.shape)

print('\nFirst 5 samples:\n', X[:5, :])
print('\nFirst 5 labels:', y[:5])

plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=40, label='1', marker='s')
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='upper right')



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                    random_state=1234, stratify=y)
print(X_train.shape)  # (75, 2)
print(y_train.shape)  # (75,)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)
print(prediction)
print(y_test)
print(np.mean(prediction == y_test))  #0.84

print(classifier.score(X_test, y_test))  #0.84

print(classifier.coef_)
print(classifier.intercept_)
plt.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', s=40, label='0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='red', s=40, label='1', marker='s')
incorrect_idx = np.where(prediction != y_test)[0]
plt.scatter(X_test[incorrect_idx, 0], X_test[incorrect_idx, 1], color='green')
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.legend(loc='upper right')
plt.show()

