'''
使用朴素贝叶斯分类器从个人广告中获取区域倾向
'''

from numpy import *
from bayes import *
from bayes_use import *
'''
分别从美国的两个城市中选取一些人，通过分析这些人发布的征婚广告信息，来比较这两个城市的人们在广告用词上是否
不同，如果结论确实不同，那么他们各自常用的词是哪些 从人们的用词当中，我们能否对不同城市的人所关心的内容有所了解
'''

# 下面将使用来自不同城市的广告训练一个分类器，然后观察分类器的效果。我们的目的不是使用该分类器进行分类，而是通过观察
# 单词和条件概率值来发现与特定城市相关的内容


#  收集数据：导入RSS源
#  RSS源分类器及高频词去除函数

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
	    freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
	import feedparser
	docList=[]
	classList=[]
	fullText=[]
	minLen = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(minLen):
		wordList = textParse(feed1['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		wordList = textParse(feed0['entries'][i]['summary'])
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)

	vocabList = createVocabList(docList)
	top30Words = calcMostFreq(vocabList, fullText)
	for pairW in top30Words:
		if pairW[0] in vocabList:
			vocabList.remove(pairW[0])

	trainingSet = list(range(2*minLen))
	testSet = []
	for i in range(20):
		randIndex = int(random.uniform(0, len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del (trainingSet[randIndex])
	trainMat=[]
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex] :
			errorCount += 1
	print("the error rate is : ", float(errorCount)/len(testSet))
	return vocabList, p0V, p1V



#  分析数据：显示地域相关的用词
#  最具表征性的词汇显示函数
def getTopWords(ny, sf):
	import operator
	vocabList, p0V, p1V = localWords(ny, sf)
	topNY=[]
	topSF=[]
	for i in range(len(p0V)):
		if p0V[i] >-6.0 : topSF.append((vocabList[i],p0V[i]))
		if p1V[i] >-6.0 : topNY.append((vocabList[i],p1V[i]))
	sortedSF = sorted(topSF, key=lambda pair:pair[1], reverse=True)
	print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
	for item in sortedSF:
		print(item[0])
	sortedNY = sorted(topNY, key=lambda pair:pair[1], reverse=True)
	print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
	for item in sortedNY:
		print(item[0])




if __name__ == '__main__':
	import feedparser
	ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
	sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
	getTopWords(ny, sf)

