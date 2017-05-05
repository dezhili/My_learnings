# # pickle(序列化模块)
# # dump()  load()

# import pickle
# my_data = {"name": "Python", "type": "Language", "version": "3.5"}
# fp = open("picklefile.dat", "wb")
# pickle.dump(my_data,fp)
# fp.close()

# fp = open("picklefile.dat","rb")
# print(pickle.load(fp))

# #JSON

# #使用traceback获取栈信息

# # import traceback
# # import sys

# # gList=['a','b','c','d']
# # def f():
# # 	gList[2]
# # 	return g()
# # def g():
# # 	gList[3]
# # 	return h()
# # def h():
# # 	print(gList[4])
# # if __name__ =='__main__':
# # 	try:
# # 		f()
# # 	except IndexError as ex:
# # 		print("Sorry,Exception occured,you accessed an element out of range")
# # 		print(ex)
# # 		traceback.print_exc()




# #使用logging记录日志信息
# import traceback
# import sys
# import logging

# gList=['a','b','c','d']
# logging.basicConfig(  #配置日志的输出方式及格式
# 	level = logging.DEBUG,
# 	filename = 'log.txt',
# 	filemode = 'w',
# 	format = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
# )

# def f():
# 	gList[2]
# 	logging.info('[INFO]:calling method g() in f()')  #记录正常的信息
# 	return g()
# def g():
# 	gList[3]
# 	logging.info('[INFO]:calling method h() in g()')
# 	return h()
# def h():
# 	print(gList[4])
# if __name__ =='__main__':
# 	logging.debug('Information during calling f():')
# 	try:
# 		f()
# 	except IndexError as ex:
# 		print("Sorry,Exception occured,you accessed an element out of range")
# 		ty,tv,tb = sys.exc_info()
		
# 		#记录异常错误信息
# 		logging.error("[ERROR]:Sorry,Exception occured,you accessed an element out of range")
# 		logging.critical('object info:%s'%ex)
		
# 		#记录异常的类型和对应的值
# 		logging.critical('Error Type:{0},Error Information:{1}'.format(ty,tv))

# 		#记录具体的trace信息
# 		logging.critical(''.join(traceback.format_tb(tb)))
# 		sys.exit(1)




# 多线程  threading
# GIL 的存在使得Python多线程编程暂时无法充分利用多处理器的优势
# threading模块基于thread进行包装，将线程的操作对象化
# threading中的join()能阻塞当前上下文环境的线程，直到调用此方法的线程终止或到达指定的timeout
# 利用该方法可以方便地控制主线程和子线程以及子线程之间的执行
#  threading 支持守护线程
#  thread模块不支持守护线程，当主线程退出的时候，所有的子线程不论是否还在工作，都会被强制结束



# import threading ,time, sys
# class test(threading.Thread):
# 	def __init__(self,name,delay):
# 		threading.Thread.__init__(self)
# 		self.name = name
# 		self.delay = delay

# 	def run(self):
# 		print('%s delay for %s'%(self.name,self.delay))
# 		time.sleep(self.delay)
# 		c = 0
# 		while True:
# 			print("This is thread %s on line %s"%(self.name,c))
# 			c = c+1
# 			if c==3:
# 				print("End of thread %s"% self.name)
# 				break

# t1 = test('Thread 1',2)
# t2 = test('Thread 2',2)
# t1.start()
# print("Wait t1 to end")
# t1.join()
# t2.start()
# print("End of main")


#  Queue 使多线程编程更安全.线程间的同步 与 互斥, 线程间数据的共享(线程安全)
# import queue
# import threading
# import random

# writelock = threading.Lock()
# class Producer(threading.Thread):
# 	def __init__(self,q,con,name):
# 		super(Producer,self).__init__()
# 		self.q = q
# 		self.con = con 
# 		self.name = name
# 		print("Producer "+self.name+" Started")

# 	def run(self):
# 		while 1:
# 			global writelock
# 			self.con.acquire()    #  获取锁对象
# 			if self.q.full():
# 				with writelock:   #输出信息
# 					print("Queue is full,producer wait!")
# 				self.con.wait()
# 			else:
# 				value = random.randint(0,10)
# 				with writelock:
# 					print(self.name+" put value "+str(value)+" into queue")
# 				self.q.put(self.name+":"+str(value))  #  放入队列中
# 				self.con.notify()  #  通知消费者
# 			self.con.release()  #  释放锁对象

# class Consumer(threading.Thread):
# 	def __init__(self,q,con,name):
# 		super(Consumer,self).__init__()
# 		self.q = q
# 		self.name = name
# 		self.con = con
# 		print("Consumer "+self.name+" Started\n")

# 	def run(self):
# 		while 1:
# 			global writelock
# 			self.con.acquire()    #  获取锁对象
# 			if self.q.empty():    #队列为空
# 				with writelock:   #输出信息
# 					print("Queue is empty,consumer wait!")
# 				self.con.wait()   #  等待资源
# 			else:
# 				value = self.q.get()  #  获取一个元素
# 				with writelock:
# 					print(self.name+" get value "+str(value)+" from queue")
				
# 				self.con.notify()  #  发送消息通知生产者
# 			self.con.release()  #  释放锁对象

# if __name__ =='__main__':
# 	q = queue.Queue(10)
# 	con = threading.Condition()  #条件变量锁
# 	p = Producer(q,con,"P1")
# 	p.start()
# 	p1 = Producer(q,con,"P2")
# 	p1.start()
# 	c1 = Consumer(q,con,"C1")
# 	c1.start()




# import os
# import queue
# import threading
# from urllib import request

# class DownloadThread(threading.Thread):
# 	def __init__(self,queue):
# 		threading.Thread.__init__(self)
# 		self.queue = queue

# 	def run(self):
# 		while True:
# 			url = self.queue.get()   #  从队列中取出一个url元素
# 			print(self.name+" begin download "+url+" ...")
# 			self.download_file(url)  #  进行文件下载
# 			self.queue.task_done()  # 下载完毕发送信号
# 			print(self.name+" download completed!!!")

# 	def download_file(self,url):
# 		urlhandler = request.urlopen(url)
# 		fname = os.path.basename(url)+".html"
# 		with open(fname,"wb") as f:              #  打开文件
# 			while True:
# 				chunk = urlhandler.read(1024)
# 				if not chunk: break
# 				f.write(chunk)

# if __name__ == '__main__':
# 	urls = ["http://wiki.python.org/moin/WebProgramming",
# 			"http://wiki.python.org/moin/Documentation",
# 			"http://tieba.baidu.com/p/2166231880"]
# 	queue = queue.Queue()

# 	#  创建一个 线程池(threading pool) ，给一个队列
# 	for i in range(5):
# 		t = DownloadThread(queue)  #  启动5个线程同时进行下载
# 		t.setDaemon(True)
# 		t.start()

# 	#  give the queue some data
# 	for url in urls:
# 		queue.put(url)

# 	# wait for the queue to finish
# 	queue.join()



#  设计模式
#  利用模块实现单例模式(23种)
#  通过单例模式可以保证系统中一个类只有一个实例而且该实例易于被外界访问，
#  从而方便对实例个数的控制并节约系统资源、





#  小甲鱼


#  内嵌函数和闭包
#  在函数内部访问全局变量可以，但试图修改全局变量，会在内部创建一个同名的局部变量
#  但如果真的想要修改，global关键字把内部创建的局部变量会变成全局变量
#  闭包是函数式编程的一个重要方面
def funx(x):
	def funy(y):
		'funy()是一个闭包'
		return x*y
	return funy
i = funx(8)

print(i)
print(type(i))
print(i(5))
print(funx(5)(5))

def funx():
	x = 5
	def funy():
		'funy()是一个闭包'
		nonlocal x
		x *= x
	return funy()
funx()





#  lambda
#  匿名函数
g = lambda x: x*3+1
h = lambda x,y : x+y
print(g(5))
print(h(3,4))





#  两个牛逼的bif
#  1. filter 返回一个可迭代对象(滤掉返回0的元素)
print(list(filter(None, [1,0,4,5])))
def odd(x):
	return x % 2
temp = range(10)
show  = filter(odd, temp)
print(list(show))

show1 = filter(lambda x : x%2 , temp)
print(list(show1))

# 2. map()
print(list(map(lambda x: x*2, range(5))))






#  递归(分治思想)
# def factorial(n):
# 	if n==1:
# 		return 1
# 	else:
# 		return n*factorial(n-1)
# number = int(input("please enter a number: "))
# result = factorial(number)
# print('%d 的阶乘是 %d'%(number, result))

def fib(n):  #  斐波那契数列
	if n<1:
		print("输入有误")
		return -1
	if n==1 or n==2:
		return 1
	else:
		return fib(n-1)+fib(n-2)
print(fib(20))

def hanoi(n, x, y, z):
	if n==1:
		print(x," ---> ",z)
	else:
		hanoi(n-1, x, z, y)	 #  将前n-1个盘子从x移动到y
		print(x," ---> ",z)  #  将最底下的那个盘子从x移动到z
		hanoi(n-1, y, x, z)  #  将y上的n-1个盘子移动到zshang
# n = int(input("请输入层数: "))
# hanoi(n, 'X', 'Y', 'Z')






#  字典(映射类型)
# brand = ['李宁', '耐克', '阿迪达斯']
# slogan = ['一切皆有可能', 'Just do it', 'Impossible is nothing']
# print('阿迪达斯的口号是 '+slogan[brand.index('阿迪达斯')])
# #  字典的内建函数
# dict1 = dict()
# dict2 = dict(((1,'li'),(2,'de'),(3,'zhi')))
# print(dict2)
# dict11 = dict1.fromkeys((1, 2, 3))
# print(dict11)
# dict12 = dict1.fromkeys((1,2,3),('one','two'))
# print(dict12)

# keys() values() items() get() in  not in(查找的是键) clear() copy(浅拷贝) pop() popitem() update()
# dict1.get(32,'木有')






#  文件
def save_file(boy, girl, count):
	file_name_boy = "boy_"+str(count)+".txt"
	file_name_girl = "girl_"+str(count)+".txt"

	boy_file = open(file_name_boy, 'w')
	girl_file = open(file_name_girl, 'w')

	boy_file.writelines(boy)
	girl_file.writelines(girl)

	boy_file.close()
	girl_file.close()

def split_file(file_name):
	f = open(file_name)
	boy=[]
	girl=[]
	count=[]

	for each_line in f:
		if each_line[:6] != '======':
			(role,line_words) = each_line.split(':',1)
			if role == '小甲鱼':
				boy.append(line_words)
			if role == '小客户':
				girl.append(line_words)
		else:
			save_file(boy, girl, count)

			boy = []
			girl = []
			count += 1

	save_file(boy,girl,count)
	f.close()




#  模块
#  os (访问文件系统)、
import os
print(os.getcwd())
# os.chdir()
print(os.listdir("D:\\"))
# os.mkdir('d:\\A')
# os.rmdir('d:\\A')
# os.system('cmd')
# os.system('calc')
# os.curdir()
# os.pardir()

# os.path 模块
# print(os.path.basename('C:\\Users\\lenovo\\Desktop\\note.py'))
# print(os.path.dirname('C:\\Users\\lenovo\\Desktop\\note.py'))
# print(os.listdir(os.path.dirname('C:\\Users\\lenovo\\Desktop\\note.py')))




#  pickle
# import pickle
# my_list = [1,2,'banana',['taimin','sunnaen']]
# pickle_file = open('my_list.pkl','wb')
# pickle.dump(my_list, pickle_file)
# pickle_file.close()

# pickle_file = open('my_list.pkl','rb')
# my_list2 = pickle.load(pickle_file)
# print(my_list2)





#  Exception(异常)
# AssertionError
# AttributeError
# IndexError
# KeyError
# NameError
# OSError FileNotFoundError 
# SyntaxError
# TypeError
# ZeroDivisionError

# try:
# 	int('abc')
# 	sum = 1+'i'
# 	f = open('s.txt')
# 	print(f.read())
# 	f.close()
# except (ValueError, OSError, TypeError):
# 	print("出错了!")

# try:
# 	f = open('f.txt','w')
# 	print(f.write('我成功了'))
# 	sum = 1+'i'
# except (OSError, TypeError):
# 	print("出错啦")
# finally:
# 	f.close()
# except OSError as reason:
# 	print("文件出错了，出错的原因是: "+str(reason))
# except TypeError :
# 	print("类型转换出错啦")






#  丰富的else语句 及 with
#  if ... else  for...else while...else(只能在循环成功后才会执行else里的语句)
#  try...except...else
# def showMaxFactor(num):
# 	count = num//2
# 	while count >1:
# 		if num % count==0:
# 			print("%d 的最大约数是 %d"%(num, count))
# 			break
# 		count -= 1
# 	else:
# 		print("%d是素数!"%num)
# num = int(input("请输入一个数: "))
# showMaxFactor(num)

# try:
# 	print(int('123'))
# except ValueError as reason:
# 	print('出错啦！'+str(reason))
# else:
# 	print("没有任何异常")




# easygui

# import easygui
# easygui.msgbox('hello let\'s go')
# import easygui as g
# import sys

# while 1:
# 	g.msgbox("嗨，欢迎进入第一个界面小游戏^_^")

# 	msg = "请问你希望在这里学到什么知识呢?"
# 	title = "小游戏互动"
# 	choices= ["谈恋爱","编程","琴棋书画","打LOL"]
# 	choice = g.choicebox(msg, title, choices)

# 	g.msgbox("你的选择是: "+ str(choice),"结果")

# 	msg = "你希望重新开始游戏吗?"
# 	title = "请选择"

# 	if g.ccbox(msg, title):
# 		pass  #选择 Continue
# 	else:
# 		sys.exit(0)  #选择 Cancel





#  类和对象
#  继承 组合 

#  self
class Ball:
	def setName(self, name):
		self.name = name
	def kick(self):
		print("该死的，我叫%s ,我在踢你"%(self.name))
b = Ball()
b.setName("球B")
b.kick()
a = Ball()
a.setName("球A")
a.kick()


#  __init__(self, param)
class Ball:
	def __init__(self, name):
		self.name = name
	def kick(self):
		print("该死的，我叫%s ,我在踢你"%(self.name))
c = Ball("土豆")
c.kick()

#  公有和私有
class Person:
	# name = "小甲鱼"
	__name = "小甲鱼"
	def getName(self):
		return self.__name
p = Person()
# print(p.__name)  #私有，查看不到
print(p.getName())
print(p._Person__name)  #  name mangling



#  继承(纵向关系)
import random as r
class Fish:
	def __init__(self):
		self.x = r.randint(0, 10)
		self.y = r.randint(0, 10)
	def move(self):
		self.x -= 1
		print("我的位置是:",self.x,self.y)
class GoldFish(Fish):
	pass
class Salmon(Fish):
	pass
class Shark(Fish):
	def __init__(self):
		# Fish.__init__(self)  #  调用未绑定的父类方法
		super().__init__()
		self.hungry = True
	def eat(self):
		if self.hungry:
			print("吃货的梦想就是天天有的吃")
		else:
			print("太撑了，吃不下了")
fish = Fish()
fish.move()
shark = Shark()
shark.move()  #  'Shark' object has no attribute 'x'




#  组合(横向关系)
class Turtle:
	def __init__(self, x):
		self.num = x
class Fish:
	def __init__(self, x):
		self.num = x
class Pool:
	def __init__(self, x, y):
		self.turtle = Turtle(x)
		self.fish = Fish(y)
	def print_num(self):
		print("水池里总共有乌龟 %d 只,小鱼 %d 只"%(self.turtle.num, self.fish.num))
pool = Pool(1, 10)
pool.print_num()

#  类 类对象(C.count) 实例对象

#  一些相关的BIF
#  issubclass(class, classinfo)  isinstance(object, classinfo)
#  hasattr(object, name)   getattr(object, name[,default])   setattr(object, name, value)
#  delattr(object, name)   property(fget=None, fset=None, fdel=None, doc=None)

class C:
	def __init__(self, x=0):
		self.x = x
c = C()
print(hasattr(c, 'x'))

print(getattr(c, 'x'))
print(getattr(c, 'y', '您所访问的属性不存在'))

setattr(c, 'y', 'LiDeZhi0')
print(getattr(c, 'y', '您所访问的属性不存在'))

# delattr(c, 'y')

# class C:
# 	def __init__(self, size=10):
# 		self.size = size
# 	def getSize(self):
# 		return self.size
# 	def setSize(self, value):
# 		self.size = value
# 	def delSize(self):
# 		del self.size
# 	x = property(getSize, setSize, delSize)
# c = C()
# print(c.x)
# c.x = 20
# print(c.size)





#  魔法方法
#  构造和析构
#  __init__()
# class Rectangle:
# 	def __init__(self, x, y):
# 		self.x = x
# 		self.y = y
# 	def getPeri(self):
# 		return 2*(self.x + self.y)
# 	def getArea(self):
# 		return self.x * self.y
# rec = Rectangle(3, 4)
# print(rec.getPeri())

# #  __new__(cls[,...])
# class CapStr(str):
# 	def __new__(cls, string):
# 		string = string.upper()
# 		return str.__new__(cls, string)
# a = CapStr("I love you")
# print(a)

# #  __del__(self) 析构器
# class C:
# 	def __init__(self):
# 		print("我是__init__方法")
# 	def __del__(self):
# 		print("我是__del__方法")
# c1 = C()
# c2 = c1
# c3 = c1
# del c2
# del c3
# del c1  #最后的引用

#  工厂函数(就是类对象) __add__ __sub__ __mul__ __mod__... 算法运算的魔法方法
print(type(len))
print(type(dir))
print(type(int))
print(type(list))
a = int("123")
class New_int(int):
	def __add__(self, other):
		return int.__sub__(self, other)
	def __sub__(self, other):
		return int.__add__(self, other)
b = New_int(3)
c = New_int(4)
print(b+c)
print(b-c)

#  反运算 __radd__ 增量赋值 __iadd__



#  类的定制(计时器)(类的方法名和属性名相同时，属性会覆盖方法)
# class C():
# 	def __str__(self):
# 		return "大道与共"
# c = C()
# print(c)


# class B():
# 	def __repr__(self):
# 		return "大道与共"
# b = B()
# b  #在idle可以直接看到
# import time
# class MyTimer():

# 	def __init__(self):
# 		self.unit = ['年','月', '天', '小时', '分钟', '秒']
# 		self.prompt = "未开始计时"
# 		self.lasted = []
# 		self.begin = 0
# 		self.end = 0

# 	def __str__(self):
# 		return self.prompt

# 	__repr__ = __str__

# 	def __add__(self, other):
# 		prompt = "总共运行了"
# 		result=[]
# 		for index in range(6):
# 			result.append(self.lasted[index] + other.lasted[index])
# 			if result[index]:
# 				prompt += (result[index]+self.unit[index])
# 		return prompt

# 	def start(self):
# 		self.begin = time.localtime()  #  元组结构 前6个
# 		self.prompt = "提示! 请先调用stop() 停止计时"
# 		print("计时开始...")

# 	def stop(self):
# 		if not self.begin:
# 			print("提示! 请先调用start() 开始计时")
# 		else:
# 			self.end = time.localtime()
# 			self._calc()
# 			print("计时结束!")

# 	def _calc(self):
# 		self.lasted = []
# 		self.prompt = "总共计时了"
# 		for index in range(6):
# 			self.lasted.append(self.end[index]-self.begin[index])
# 			if self.lasted[index]:
# 				self.prompt += (str(self.lasted[index]) + self.unit[index])

# 		self.begin = 0
# 		self.end = 0

# t1 = MyTimer()
# t1.stop()
# t1.start()
# t1.stop()
# print(t1)


#  属性访问(魔法方法)  #  容易出现死循环
#  __getattr__(self, name) -试图获取一个不存在的属性
#  __getattribute__(self, name)
#  __setattr__(self, name, value)  #  赋值操作
#  __delattr__(self, name)

# class C:
# 	def __getattribute__(self, name):
# 		print("getattribute")
# 		return super().__getattribute__(name)
# 	def __getattr__(self, name):
# 		print("getattr")
# 	def __setattr__(self, name, value):
# 		print("setattr")
# 		super().__setattr__(name, value)
# 	def __delattr__(self, name):
# 		print("delattr")
# 		super().__delattr__(name)
# c = C()
# print(c.x)
# c.x = "setting"
# print(c.x)
# del c.x

class Rectangle:
	def __init__(self, width=0, height=0):
		self.width = width
		self.height = height
	def __setattr__(self, name, value):
		if name == 'square':
			self.width = value
			self.height = value
		else:
			super().__setattr__(name, value)  #  不会出现死循环
			#  self.__dict__[name]=value
	def getArea(self):
		return self.width * self.height
r1 = Rectangle(4,5)
print(r1.getArea())
r1.square = 10
print(r1.getArea())
print(r1.__dict__)




#  魔法防法:描述符(Property)
#  描述符就是将某种特殊类型的类的实例指派给另一个类的属性
#  __get__(self, instance, owner) 用于访问属性，返回属性的值
#  __set__(self, instance, value) 将在属性分配操作中调用， 不返回任何内容
#  __delete__(self, instance)

# class MyDescripter:
# 	def __get__(self, instance, owner):
# 		print("getting...", self, instance, owner)
# 	def __set__(self, instance, value):
# 		print("setting...", self, instance, value)
# 	def __delete__(self, instance):
# 		print("deleting...", self, instance)	
# class Test:
# 	x = MyDescripter()  #  MyDescripter 就是属性x的描述符
# test = Test()
# print(test.x)
# test.x = "X-man"
# print(test.x)
# del test.x

#  练习: 定义一个温度类，然后定义两个描述符类用于描述摄氏度和华氏度两个属性，自动转换
class Celsus:
	def __init__(self, value=28.0):
		self.value  =float(value)
	def __get__(self, instance, owner):
		return self.value
	def __set__(self, instance, value):
		self.value = float(value)
class Fahrea:
	def __get__(self, instance, owner):
		return instance.cel * 1.8 + 32
	def __set__(self, instance, value):
		instance.cel = (float(value)-32)/1.8

class Temperature:
	cel = Celsus()
	fah = Fahrea()
temp = Temperature()
print(temp.cel)
temp.cel = 30
print(temp.cel)
print(temp.fah)
temp.fah = 100
print(temp.cel)



#  定制容器 定制序列
#  如果希望定制的容器是不可变的，只需定义__len__() __getitem__()
#  如果是可变的，需定义 __len__() __getitem__() __setitem__() __delitem__()

#  不可变
class CountList:
	def __init__(self, *args):
		self.values = [x for x in args]
		self.count = {}.fromkeys(range(len(self.values)), 0)
	def __len__(self):
		return len(self.values)
	def __getitem__(self, key):
		self.count[key]+=1
		return self.values[key]
c1 = CountList(1, 3, 5, 7, 9)
print(c1[1])
print(c1.count)



#  迭代器
#  bif  iter() next()
#  魔法方法 __iter__()  __next__() 
# string = "lidezhi"
# it = iter(string)
# print(next(it)) 

class Fibs:
	def __init__(self, n=10):
		self.s = 0
		self.a = 0
		self.n = n
	def __iter__(self):
		return self
	def __next__(self):
		self.s, self.a = self.a, self.s+self.a
		if self.a > self.n:
			raise StopIteration
		return self.a
fibs = Fibs()
for each in fibs:
	print(each)
# fibs = Fibs(100)
# for each in fibs:
# 	print(each)  





