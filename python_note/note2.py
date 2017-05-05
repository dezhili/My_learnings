#  迭代器
#  bif  iter() next()
#  魔法方法 __iter__()  __next__() 
# string = "lidezhi"
# it = iter(string)
# print(next(it)) 

# class Fibs:
# 	def __init__(self, n=10):
# 		self.s = 0
# 		self.a = 0
# 		self.n = n
# 	def __iter__(self):
# 		return self
# 	def __next__(self):
# 		self.s, self.a = self.a, self.s+self.a
# 		if self.a > self.n:
# 			raise StopIteration
# 		return self.a
# fibs = Fibs()
# for each in fibs:
# 	print(each)
# fibs = Fibs(100)
# for each in fibs:
# 	print(each)  




# 生成器(是迭代器的一种实现，只需要在普通函数中增加一个yield语句 )
# 协同程序就是可以运行的独立函数调用，函数可以暂停或挂起，
# 并在需要的时候从离开的地方继续或重新开始
def myGen():
	print("生成器被执行")
	yield 1
	yield 2
myG = myGen()
print(next(myG))
print(next(myG))

def fibs():
	a = 0
	b = 1
	while True:
		a, b = b, a+b
		yield a
for ele in fibs():
	if ele > 100:
		break
	print(ele, end=' ')
#列表推导式
list1 = [i for i in range(100) if not (i%2) and i%3]
print(list1)
#字典推导式
dict1 = {i:i%2==0 for i in range(10)}
print(dict1)
#集合推导式
set1 = {i for i in [1,1,2,3,4,4,4,5,5,5,6]}
print(set1)

##生成器推导式
e = (i for i in range(10)) 
print(e)
print(next(e))
print(sum(i for i in range(100) if i % 2))




#  模块 -> 程序
#容器 -> 数据的封装  函数 -> 语句的封装  类 -> 方法和属性的封装
#  命名空间
#  导入模块

#  if __name__ == '__main__'
#  搜索路径
# import sys
# print(sys.path)
# sys.path.append("F:\\tensorflow学习\\site-packages")
# print(sys.path)

#  包(得有__init__.py)

#  python电池 标准库
#  print(timeit.__doc__) dir(timeit) 
#  timeit.__all__(一般导入模块的时候只有__all__里面的)['Timer', 'timeit', 'repeat', 'default_timer']
