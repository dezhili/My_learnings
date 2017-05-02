from math import log
import operator

def createDataSet():
	dataSet = [[1, 1, 'yes'],
			   [1, 1, 'yes'],
			   [1, 0, 'no'],
			   [0, 1, 'no'],
			   [0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

#  计算给定数据集的香农熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1

	shannonEnt = 0.0
	for key in labelCounts:
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


#  按照给定特征划分数据集
#  参数: 待划分的数据集  划分数据集的特征  需要返回的特征的值
#  splitDataSet(myDat,0,1)--> [[1,'yes'],[1,'yes'],[0,'no']]  (myDat,0,0)->...
def splitDataSet(dataSet, axis, value): 
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


#  选择最好的数据集划分方式即最好的特征
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0])-1  #数据的最后一列或每个实例的最后一个元素是当前实例的类别标签
	baseEntropy = calcShannonEnt(dataSet)
	bestInfoGain = 0.0; bestFeature = -1

	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)

		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet, i, value)
			prob = len(subDataSet)/float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		infoGain = baseEntropy - newEntropy

		if infoGain>bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature



#  递归构建决策树
#  递归结束的条件:程序遍历完所有划分数据集的属性 或 每个分支下的所有实例都具有相同的分类

#  数据集已经处理完所有属性，但是类标签依然不是唯一的，采用 多数表决
def majorityCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(), \
							key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

#  创建树
def createTree(dataSet, labels):
	classList = [example[-1] for example in dataSet]

	# 类别完全相同则停止继续划分
	if classList.count(classList[0]) == len(classList):
		return classList[0]

	#  遍历完所有特征时返回出现次数最多的类别
	if len(dataSet[0]) == 1 :
		return majorityCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)
	bestFeatLabel = labels[bestFeat]
	myTree = {bestFeatLabel:{}}
	del labels[bestFeat]

	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet\
										(dataSet, bestFeat, value), subLabels)
	return myTree
	




#  测试和存储分类器
#  如何利用决策树执行数据分类
#  在执行数据分类时，需要使用决策树以及用于构造决策树的标签向量。然后，程序比较测试数据与决策树上的数值，递归
#执行该过程直到进入叶子节点；最后将测试数据定义为叶子节点所述的类型

def classify(inputTree, featLabels, testVec):
	firstSlides = list(inputTree.keys())
	firstStr = firstSlides[0]
	secondDict = inputTree[firstStr]

	featIndex = featLabels.index(firstStr)  #'no surfacing'在featLabels的索引
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else:
				classLabel = secondDict[key]
	return classLabel

	
#  使用算法：决策树的存储(使用python模块pickle序列化对象，
#  序列化对象可以在磁盘上保存对象，任何对象都可以执行序列化操作)

#  使用pickle模块存储决策树
def storeTree(inputTree, filename):
	import pickle
	fw = open(filename, 'w')
	pickle.dump(inputTree,fw)
	fw.close()
def grabTree(filename):
	import pickle
	fr = open(filename,'r')
	return pickle.load(fr)


