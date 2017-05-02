#通用的学习模式
import numpy as np 
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data 
iris_Y = iris.target

# print(iris_X[:2,:])
# print(iris_Y[:51])

X_train,X_test,y_train,y_test = train_test_split(
	iris_X,iris_Y,test_size=0.3)
# print(y_train)
# print(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
print(knn.predict(X_test))
print(y_test)






