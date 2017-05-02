from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO

data = open(r"C:\Users\lenovo\Desktop\a.csv",'r')
reader = csv.reader(data)
headers = reader.__next__()
print(headers)


###data preprocessing
featureList=[]
labelList = []

for row in reader:
    labelList.append(row[len(row)-1])
    rowDict = {}#DictVectorizer 特征值放入
    for i in range(1,len(row)-1):
        rowDict[headers[i]] = row[i]

    featureList.append(rowDict)
print(featureList)

#Vectorize features
vec = DictVectorizer()#sklearn 将字典里的数据直接转化为0101
dummyX = vec.fit_transform(featureList).toarray()
print("dummyX:"+ str(dummyX))
print(vec.get_feature_names())


print("labelList: "+ str(labelList))

#Vectorize class Labels
lb = preprocessing.LabelBinarizer()
dummyY=lb.fit_transform(labelList)
print("dummyY:"+ str(dummyY))


###using decision tree for classification
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX,dummyY)
print('clf:'+ str(clf))
print('------------------------------------------')

#visualize model
with open(r'C:\Users\lenovo\Desktop\datainformationmodel.dot','w')as f:
    f= tree.export_graphviz(clf,feature_names = vec.get_feature_names(),out_file=f)

    
###prediction
oneRowX = dummyX[0, :]
print('oneRowX:' + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print('newRowX:' + str(newRowX))

predictedY = clf.predict(newRowX)
print("predictedY:" + str(predictedY))



    

    








         





