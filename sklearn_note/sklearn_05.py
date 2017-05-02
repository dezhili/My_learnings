#cross_validation  交叉验证
#  model selection and params selection


#  基础验证
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
X =  iris.data 
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=4)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

print(knn.score(X_test,y_test))  # 0.973684210526


#  交叉验证法
from sklearn.cross_validation import cross_val_score

# scores = cross_val_score(knn,X,y,cv=5,scoring='accuracy')  # for classification
# print(scores)
# print(scores.mean())


#选择不同模型参数(k值)
import matplotlib.pyplot as plt 
k_range = range(1,31)
k_scores = []

for k in k_range:
	knn = KNeighborsClassifier(n_neighbors=k)
	# loss = -cross_val_score(knn,X,y,cv=10,scoring='mean_squared_error') # regression
	scores = cross_val_score(knn,X,y,cv=10,scoring='accuracy')  # for classification
	k_scores.append(scores.mean())

plt.plot(k_range,k_scores)
plt.xlabel('Value of k for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()