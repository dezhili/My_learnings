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
        a, b = b, a + b
        yield a
for ele in fibs():
    if ele > 100:
        break
    print(ele, end=' ')
# 列表推导式
list1 = [i for i in range(100) if not (i % 2) and i % 3]
print(list1)
# 字典推导式
dict1 = {i: i % 2 == 0 for i in range(10)}
print(dict1)
# 集合推导式
set1 = {i for i in [1, 1, 2, 3, 4, 4, 4, 5, 5, 5, 6]}
print(set1)

# 生成器推导式
e = (i for i in range(10))
print(e)
print(next(e))
print(sum(i for i in range(100) if i % 2))


#  模块 -> 程序
# 容器 -> 数据的封装  函数 -> 语句的封装  类 -> 方法和属性的封装
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


#***%% 爬虫 %%***
# 论一只爬虫的自我修养(urllib)

# url: protocol://hostname[:port]/path/[;parameters][?query]#fragment
# import urllib.request
# response = urllib.request.urlopen("http://www.fishC.com")
# html = response.read()  #bytes
# html = html.decode('utf-8') #解码
# print(html)


#  实战(猫奴)
# import urllib.request

# request = urllib.request.Request('http://placekitten.com/g/400/500')
# response = urllib.request.urlopen(request)
# response = urllib.request.urlopen('http://placekitten.com/g/400/500')#字符串或request对象
# response对象 有geturl() info() getcode()
# print(response.geturl())
# print(response.info())
# print(response.getcode())

# cat_image = response.read()
# with open("cat_400_500.jpg",'wb') as f:
# 	f.write(cat_image)


#  利用有道词典进行翻译(POST GET)
# import urllib.request
# import urllib.parse  # for parsing urls(解析)
# import json
# import time  #  延迟(防止ip访问频率过多)

# while True:
# 	contents = input("请输入要翻译的内容(输入'q!'退出): ")
# 	if contents == 'q!':
# 		break

# 	url = "http://fanyi.youdao.com/translate_o?smartresult=dict&smartresult=rule&sessionFrom=dict2.index"

# 	head={}
# 	head['User-Agent'] = "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36"

# 	# data -form data 字典
# 	# data should be a buffer in the standard application/x-www-form-urlencoded format
# 	# urllib.parse.urlencode()
# 	data = {}
# 	data['i'] = contents  #  暂时只能翻译 将军令
# 	data['from']= 'AUTO'
# 	data['to']= 'AUTO'
# 	data['smartresult']='dict'
# 	data['client']='fanyideskweb'
# 	data['salt']= '1493970954156'  #  会变
# 	data['sign'] = '9e4fb723d3e9eb6e4b43b891afe73d94'  #  会变
# 	data['doctype'] = 'json'
# 	data['version']= '2.1'
# 	data['keyfrom']= 'fanyi.web'
# 	data['action'] = 'FY_BY_CLICKBUTTON'
# 	data['typoResult'] = 'true'


# 	data = urllib.parse.urlencode(data).encode('utf-8')

# 	req = urllib.request.Request(url, data, head)  # 1. 隐藏 --修改headers 有两个途径(add_header() )
# 	response = urllib.request.urlopen(req)
# 	print(req.headers)
# 	html = response.read().decode('utf-8')
# 	print(html)  #  返回字符串 内含json格式

# 	# target = json.loads(html)  #  返回 字典格式
# 	# print("翻译结果是 %s"%(target['translateResult'][0][0]['tgt']))
# 	time.sleep(5)


# 2. 代理 --多个ip访问(防止同一ip访问频率过多)
# ① proxy_support = urllib.request.ProxyHandler({'类型':'代理ip:端口号'})
# ② 定制 创建 opener = urllib.request.build_opener(proxy_support)
# ③ 安装opener  urllib.request.install_opener(opener) 或 opener.open(url)
# import urllib.request
# import random
# url = "http://www.whatismyip.com.tw"
# iplist = ['61.191.41.130:80', '211.84.193.197:8998', '222.74.225.231:3128']

# ip_choice = random.choice(iplist)
# print(ip_choice)
# proxy_support = urllib.request.ProxyHandler({'http': ip_choice})
# opener = urllib.request.build_opener(proxy_support)
# opener.addheaders = [
#     ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')]
# urllib.request.install_opener(opener)
# response = urllib.request.urlopen(url)
# html = response.read().decode('utf-8')
# print(html)


#  实战OOXX
# import urllib.request
# import os
# import random

# def url_open(url):
# 	req = urllib.request.Request(url)
# 	req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.101 Safari/537.36')
# 	# iplist = ['61.191.41.130:80', '211.84.193.197:8998', '222.74.225.231:3128']
# 	# ip_choice = random.choice(iplist)
# 	# proxy_support = urllib.request.ProxyHandler({'http': ip_choice})
# 	# opener = urllib.request.build_opener(proxy_support)
# 	# urllib.request.install_opener(opener)

# 	response = urllib.request.urlopen(url)
# 	html = response.read()
# 	return html

# def get_page(url):
#     html = url_open(url).decode('utf-8')
#     # print(html)
#     a = html.find("current-comment-page")+23
#     b = html.find("]", a)
#     print(html[a:b])
#     return html[a:b]

# def find_images(url):
#     html = url_open(url).decode('utf-8')
#     a = html.find("img src=")
#     img_addrs = []

#     while a!= -1:
#         b = html.find(".jpg",a,a+255)
#         if b!= -1:
#             img_addrs.append("http:"+html[a+9:b+4])
#         else:
# 	        b = a+9
#         a = html.find("img src=",b)  

#     for each in img_addrs:
#         print(each)
#     return img_addrs

# def save_images(folder, img_addrs):
#     for each in img_addrs:
#     	filename = each.split('/')[-1]
#     	with open(filename, 'wb') as f:
#     		img = url_open(each)
#     		f.write(img)


# def downloadMM(folder='OOXX2', pages=10):
#     os.mkdir(folder)
#     os.chdir(folder)

#     url = "http://jandan.net/ooxx/"
#     page_num = int(get_page(url))

#     for i in range(pages):
#         page_num -= i
#         page_url = url + 'page-' + str(page_num) + '#comments'
#         img_addrs = find_images(page_url)
#         save_images(folder, img_addrs)
# if __name__ == '__main__':
# 	downloadMM()


#  正则表达式
import re
a = re.search(r'FishC', "I love FishC.com") #search()在字符串中搜索正则表达式第一次出现的位置
print(a)  #<_sre.SRE_Match object; span=(7, 12), match='FishC'>
b = re.search(r'.', 'I love Fishc.com') # .通配符 匹配除换行之外的所有字符
print(b)  #<_sre.SRE_Match object; span=(0, 1), match='I'>
c = re.search(r'Fish.', 'I love Fishc.com')
print(c)
d = re.search(r'\.', 'I love Fishc.com') # 匹配 . 本身
print(d)

e = re.search(r'\d', 'I love 123 FishC.com') # \d 匹配任何数字
print(e)
f = re.search(r'\d\d\d', 'I love 123 FishC.com') # \d 匹配任何数字
print(f)
g = re.search(r'\d\d\d\.\d\d\d\.\d\d\d\.\d\d\d', '192.168.168.1')
print(g)

h = re.search(r'[aeiou]', 'I love FishC.com') #加[]为字符类  ， 匹配
print(h)
j = re.search(r'[a-z]', 'I love FishC.com')
print(j)

k = re.search(r'ab{3}c', 'abbbc')			#限定重复匹配的次数{}
print(k)										
l = re.search(r'ab{3,10}c', 'abbbbbc')  
print(l)

m = re.search(r'[01]\d\d|2[0-4]\d|25[0-5]', '188')  	#匹配0-255  逻辑或
print(m)

#匹配ip地址 , 192.168.001.001
n = re.search(r'(([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])\.){3}([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])','192.168.1.1')   
print(n) 
   