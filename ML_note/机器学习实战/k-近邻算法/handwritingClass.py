from numpy import *
from os import listdir
import operator


#  准备数据：将图像转换为测试向量
#  32x32 -》 1x1024
def img2vector(filename):
	returnVect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect


#  测试算法
def handwritingClassTest():
	hwLabels=[]

	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))

	for i in range(m):
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)

	testFileList = listdir('testDigits')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d"\
							%(classifierResult, classNumStr))
		if(classifierResult != classNumStr):
			errorCount += 1.0
	print("\nthe total number of errors is: %d"%errorCount)
	print("\nthe total error rate is %f"%(errorCount/float(mTest)))



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


if __name__ == '__main__':
	handwritingClassTest()