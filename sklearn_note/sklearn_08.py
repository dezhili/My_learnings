# 保存模型
from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X,y = iris.data,iris.target
clf.fit(X,y)

#pickle
# import pickle
# with open('save\clf.pickle','wb') as f:
# 	pickle.dump(clf,f)

# with open('save\clf.pickle','rb') as f:
# 	clf2 = pickle.load(f)
# 	print(clf.predict(X[0:1]))


# joblib
from sklearn.externals import joblib

# joblib.dump(clf,'save\clf.pkl')

clf3 = joblib.load('save\clf.pkl')
print(clf3.predict(X[0:1]))