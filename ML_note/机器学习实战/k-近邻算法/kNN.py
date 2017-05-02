#  使用python导入数据
from numpy import *
import operator

# def createDataSet():
# 	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
# 	labels = ['A','A','B','B']
# 	return group, labels


#  实施kNN分类算法
def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX, (dataSetSize,1)) - dataSet  #按列重复组
	sqDiffMat = diffMat ** 2  
	sqDistances = sqDiffMat.sum(axis=1)  #按行相加
	distances = sqDistances ** 0.5
	sortedDistIndicies = distances.argsort()  #  将元素从小到大按顺序排列提取对应索引 [1,2,0]

	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) +1
	sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #  返回由tuple组成的列表
	return sortedClassCount[0][0]


'''
使用k近邻算法改进约会网站的配对效果

'''

#准备数据：从文本文件中解析数据
#输入为文件名字符串，输出为训练样本矩阵和类标签向量
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector



#分析数据：使用Matplotlib创建散点图
import matplotlib
import matplotlib.pyplot as plt 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
# ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels), 15.0*array(datingLabels))
# plt.show()





#准备数据：归一化数值(处理这种不同取值范围的特征值，处理为0-1或-1-1)  newValue = （oldValue-min）/(max-min)
#归一化特征值
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals-minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet/tile(ranges, (m,1))
	return normDataSet, ranges, minVals


#测试算法:作为完整程序验证分类器
def datingClassTest():
	hoRatio = 0.10
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

	
	normMat, ranges, minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],\
							datingLabels[numTestVecs:m], 3)
		print("the classifier came back with: %d, the real answer is: %d"\
							%(classifierResult, datingLabels[i]))
		if(classifierResult != datingLabels[i]):
			errorCount += 1.0
	print("the total error rate is: %f"%(errorCount/float(numTestVecs)))


#使用算法: 构建完整可用系统
#约会网站预测函数
def classifyPerson():
	resultList = ['not at all', 'in some doses', 'in large doses']
	percentTats = float(input("percentage of time spent playing video games? "))
	ffMiles = float(input("frequent flier miles earned per year? "))
	iceCream = float(input("liters of ice cream consumed per year? "))

	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')

	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles, percentTats, iceCream])
	classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
	print("You will probably like this person : ", resultList[classifierResult-1]) #假设3代表in large doses

	fig = plt.figure()
	ax = fig.add_subplot(111)
	# ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
	ax.scatter(datingDataMat[:,1], datingDataMat[:,2],15.0*array(datingLabels), 15.0*array(datingLabels))
	plt.show()






if __name__ == '__main__':
	# datingClassTest()
	classifyPerson()






