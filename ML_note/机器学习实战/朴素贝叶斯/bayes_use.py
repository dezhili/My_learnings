'''
使用朴素贝叶斯过滤垃圾邮件
'''
from numpy import *
from bayes import *



#  准备数据：切分文本
# mySent='The book is the best book on Python or M.L. I have ever laid eyes .'
# import re
# regex = re.compile('\\W*')
# listOfTokens = regex.split(mySent)

# [tok for tok in listOfTokens if len(tok)>0]
# [tok.lower() for tok in listOfTokens if len(tok) >0]

# print(listOfTokens)
# emailText = open('6.txt').read()
# listOfTokens = regex.split(emailText)  #  文本解析



#  测试算法:使用朴素贝叶斯进行交叉验证

#  文件解析 及 完整的垃圾邮件测试函数
def textParse(bigString):
	import re
	listOfTokens = re.split(r'\W*', bigString)
	return [tok.lower() for tok in listOfTokens if len(tok)>2]

def spamTest():
	docList = []
	classList = []
	fullText = []
	for i in range(1,26):
		wordList = textParse(open('email/spam/%d.txt'%i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		wordList = textParse(open('email/ham/%d.txt'%i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	vocabList = createVocabList(docList)
	
	trainingSet = list(range(50))
	testSet = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])

	trainMat = [];
	trainClasses = []
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
		trainClasses.append(classList[docIndex])

	p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
	errorCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errorCount += 1
	print('the error rate is : ', float(errorCount)/len(testSet))

if __name__ == '__main__':
	spamTest()




