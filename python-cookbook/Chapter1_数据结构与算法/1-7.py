'''
1.7 字典排序
创建一个字典，并且在迭代或序列化这个字典的时候能够控制元素的顺序
collections.OrderedDict， 在迭代操作的时候它会保持元素被插入时的顺序。
OrderedDict 内部维护着一个根据键插入顺序排序的双向链表。每次当一个新的元素插入进来的时候，它会被放到
链表的尾部。对于一个已经存在的键的重复赋值不会改变键的顺序。
'''
from collections import OrderedDict
d = OrderedDict()
d['foo'] = 1
d['bar'] = 2
d['spam'] = 3
d['grok'] = 4

for key in d:
    print(key, d[key])


# 当想要构建一个将来需要序列化或编码成其他格式的映射的时候，OrderedDict is useful
import json
j = json.dumps(d)
print(j)


'''
1.8 字典的运算
怎样在数据字典中执行一些计算操作(比如求最小值、 最大值、 排序等等)
'''
prices = {
    'ACME': 45.23,
    'AAPL': 612.78,
    'IBM': 205.55,
    'HPQ': 37.20,
    'FB': 10.75
}

min_price = min(zip(prices.values(), prices.keys()))
print(min_price)
max_price = max(zip(prices.values(), prices.keys()))
print(max_price)

prices_sorted = sorted(zip(prices.values(), prices.keys()))

# zip() 创建的是一个只能访问一次的迭代器。

# min(prices)  #得到的是在字典的键上的操作
# min(prices.values())  #如果还想要知道对应的键的信息

# min(prices, key=lambda k: prices[k])  # Returns 'FB'
# min_value = prices[min(prices, key=lambda k: prices[k])]


# zip() 通过将字典"反转"为(值， 键)元组序列来解决了上述问题
