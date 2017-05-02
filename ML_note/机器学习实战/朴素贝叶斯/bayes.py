'''
使用Python进行文本分类

文本--》词向量 --》条件概率 --》分类器

'''
from numpy import *

#  准备数据：从文本中构建词向量
#  词表到向量的转换函数
def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
             ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
             ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
             ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
             ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
             ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1]
	return postingList,classVec

#  创建词汇表
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)

#  转换成 101010000000(我们将每个词的出现与否作为一个特征 被描述为词集模型)
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)]=1
		else:
			print("the word : %s is not in my Vocabulary"%word)
	return returnVec
#  如果该词在文档中出现不止一次，这可能意味着包含该词是否出现在文档中所不能表达的某种信息，被称为词袋模型
def bagOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec






#  训练算法：从词向量计算概率
#  p(Ci|W) = (p(W|Ci)p(Ci))/p(W)


#  朴素贝叶斯分类器训练函数
#  trainMatrix-[[0,0,...1],[1,0,...]...]-文档矩阵(词向量的集合)
#  trainCategory-[0,1,0,1,0,1] -每篇文档的类别标签构成的向量
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)
	numWords = len(trainMatrix[0])
	pAbusive = sum(trainCategory)/float(numTrainDocs)  #P(C1)文档属于侮辱性文档的概率

    #  初始化概率(初始化程序中的)(P(Wi|C1) P(Wi|C0))，防止为0改一下
	# p0Num = zeros(numWords)
	# p1Num = zeros(numWords)
	# p0Denom = 0.0
	# p1Denom = 0.0
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0
	
	for i in range(numTrainDocs):
		if trainCategory[i]==1:
			p1Num += trainMatrix[i]
			p1Denom += sum(trainMatrix[i])
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# p1Vect = p1Num/p1Denom
	# p0Vect = p0Num/p0Denom
	p1Vect = log(p1Num/p1Denom)  #p(W0|1)*p(W1|1)... 求对数可以避免下溢出
	p0Vect = log(p0Num/p0Denom)

	return p0Vect,p1Vect,pAbusive




#  测试算法:根据现实情况修改分类器
#  朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vect, p1Vect, pClass1):
	p1 = sum(vec2Classify * p1Vect) + log(pClass1)
	p0 = sum(vec2Classify * p0Vect) + log(1.0-pClass1)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPosts, listClasses = loadDataSet()
	myVocabList = createVocabList(listOPosts)

	#trainMatrix
	trainMat = []
	for postinDoc in listOPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))


	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
	print(testEntry, " classified as : ",classifyNB(thisDoc, p0V, p1V, pAb))



if __name__ == '__main__':
	# listOPosts, listClasses = loadDataSet()
	# myVocabList = createVocabList(listOPosts)

	# #trainMatrix
	# trainMat = []
	# for postinDoc in listOPosts:
	# 	trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

	# p0V, p1V, pAb = trainNB0(trainMat, listClasses)
	# print(pAb, end='\n\n')
	# print(p0V, end='\n\n')
	# print(p1V)
	# testingNB()

	