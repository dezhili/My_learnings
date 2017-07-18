'''
1.17 从字典中提取子集 
'''
#  字典推导
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}
p1 = {key : value for key,value in prices.items() if value > 200}
print(p1)
tech_names = {'AAPL', 'IBM', 'HPQ', 'MSFT'}
p2 = {key : value for key,value in prices.items() if key in tech_names}
print(p2)

#  通过创建一个元组序列然后把它传给dict()函数也能实现。
p3 = dict((key, value) for key,value in prices.items() if value > 200)
print(p3)

p2_re = {key: prices[key] for key in prices.keys() & tech_names}
print(p2_re)



'''
1.18 映射名称到序列元素
你有一段通过下标访问列表或者元组中元素的代码，
但是这样有时候会使得你的代码难以阅读， 于是你想通过名称来访问元素。
collections.namedtuple() 使用一个普通的元组对象解决这个问题。
这个函数实际上是一个返回Python中标准元组类型子类的一个工厂方法。 
你需要传递一个类型名和你需要的字段给它，然后它就会返回一个类，你可以初始化这个类，为你定义的字段传递值等。
给它，然后他就会返回一个类，你可以初始化这个类，为你定义的字段传递值
'''
from collections import namedtuple
Subscriber = namedtuple('Subscriber', ['addr', 'joined'])
sub = Subscriber('jonesy@example.com', '2017-6-29')
print(sub)
print(sub.addr)
print(sub.joined)

# 尽管 namedtuple 的实例看起来像一个普通的类实例，但是它跟元组类型是可交换的，
# 支持所有的普通元组操作，比如索引和解压。
print(len(sub))
addr, joined = sub
print(addr)
print(joined)

#  命名元组的一个主要用途是将你的代码从下标操作中解脱出来。

#  普通元组
def compute_cost(records):
    total = 0.0
    for rec in records:
        total += rec[1] * rec[2]
    return total

#  命名元组
from collections import namedtuple
Stock = namedtuple('Stock', ['name', 'shares', 'price'])
def compute_cost(records):
    total = 0.0
    for rec in records:
        s = Stock(*rec)
        total += s.shares * s.price
    return total



# 命名元组另一个用途就是作为字典的替代，因为字典存储需要更多的内存空间。
# 一个命名元组是不可更改的。
s = Stock('ACME', 100, 123.45)
print(s)
# s.shares = 75 AttributeError: can't set attribute
# 改变属性的值， 可用命名元组实例的_replace()， 它会创建一个新的命名元组并将对应的字段用新的值取代
s1 = s._replace(shares=75)
print(s1)


# _replace()是一个方便的填充数据的方法。可以先创建一个包含缺省值的原型元组，然后使用_replace()创建
# 新的值被更新过的实例。
Stock = namedtuple('Stock', ['name', 'shares', 'price', 'date', 'time'])

stock_prototype = Stock('', 0, 0.0, None, None)

# Function to convert a dictionary to a Stock
def dict_to_stock(s):
    return stock_prototype._replace(**s)

a = {'name':'ACME', 'shares':100, 'price':123.45}
print(dict_to_stock(a))
b = {'name': 'ACME', 'shares': 100, 'price': 123.45, 'date': '12/17/2012'}
print(dict_to_stock(b))
