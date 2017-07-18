'''
1.19 转换并同时计算数据
你需要在数据序列上执行聚集函数(比如 sum() , min() , max() )， 但是首先你需要先转换或者过滤数据
'''

# 使用一个生成器表达式参数
nums = [1, 2, 3, 4, 5]
s = sum(x * x for x in nums)
print(s)

# Determine if any .py files exist in a directory
# import os
# files = os.listdir('dirname')
# if any(name.endswith('.py') for name in files):
#     print('There be python !')
# else:
#     print('Sorry, no python')
# Output a tuple as csv
s = ('ACME', 50, 123.45)
print(','.join(str(x) for x in s))
# Data reduction across fields of a data structure
portfolio = [
    {'name':'GOOG', 'shares': 50},
    {'name':'YHOO', 'shares': 75},
    {'name':'AOL', 'shares': 20},
    {'name':'SCOX', 'shares': 65}
]
min_shares = min(s['shares'] for s in portfolio)
print(min_shares)


# 使用一个生成器表达式作为参数会比先创建一个临时列表更加高效和优雅。
# 生成器方案会以迭代的方式转换数据，因此更省内存
min_shares = min(portfolio, key=lambda s : s['shares'])
print(min_shares)



'''
1.20 合并多个字典或映射
现在有多个字典或者映射，你想将它们从逻辑上合并为一个单一的映射后执行某些操作， 
比如查找值或者检查某些键是否存在。
ChainMap
'''
a = {'x': 1, 'z': 3 }
b = {'y': 2, 'z': 4 }
from collections import ChainMap
c = ChainMap(a, b)
print(c['x'])
print(c['y'])
print(c['z'])

# 一个ChainMap接受多个字典并将它们在逻辑上变为一个字典。
# 然后，这些字典并不是真的合并在一起了， 
# ChainMap 类只是在内部创建了一个容纳这些字典的列表 并重新定义了一些常见的字典操作来遍历这个列表。
print(len(c))
print(list(c.keys()))
print(list(c.values()))

# 对于字典的更新或删除操作总是影响的是列表中第一个字典。
c['z'] = 10
c['w'] = 40
del c['x']
print(a)
# del c['y']  KeyError: "Key not found in the first mapping: 'y'"

# ChainMap 对于编程语言中的作用范围变量(比如 globals , locals 等)是非常有用的。 
# 事实上，有一些方法可以使它变得简单：
values = ChainMap()
values['x'] = 1
values = values.new_child()
values['x'] = 2
values = values.new_child()
values['x'] = 3
print(values)
print(values['x'])


# update() 将两个字典合并
a = {'x': 1, 'z': 3}
b = {'y': 2, 'z': 4}
merged = dict(b)
merged.update(a)
for key,values in merged.items():
    print(key, values)

# update()需要创建一个完全不同的字典对象(或破坏现有字典结构)
# 同时如果原字典做了更新，这种改变不会反映到新的合并字典中去。
# ChainMap()使用原来的字典，它自己不创建新的字典。所以它并不会产生上面所说的结果

