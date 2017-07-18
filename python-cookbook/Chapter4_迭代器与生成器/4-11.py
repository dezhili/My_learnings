'''
4.11 同时迭代多个序列
你想同时迭代多个序列, 每次分别从一个序列中取一个元素
'''
# zip(a, b) 会生成一个可返回元组(x, y)的迭代器
xpts = [1, 5, 4, 2, 10, 7]
ypts = [101, 78, 37, 15, 62, 99]
for x, y in zip(xpts, ypts):
    print(x, y)

# itertools.zip_longest()
a = [1, 2, 3]
b = ['w', 'x', 'y', 'z']
from itertools import zip_longest
for i in zip_longest(a, b):
    print(i)  #(None, 'z)

for i in zip_longest(a, b, fillvalue=0):
    print(i)  #(0, 'z')

# zip() 打包并生成一个字典
headers = ['name', 'shares', 'price']
values = ['ACME', 100, 490.1]
s = dict(zip(headers, values))
print(s)  #{'price': 490.1, 'shares': 100, 'name': 'ACME'}

# zip() 会创建一个迭代器作为结果返回。如果需要将结对的值存储在列表中，使用list() 函数
print(zip(xpts, ypts))  #<zip object at 0x000001B9774221C8>
print(list(zip(xpts, ypts)))  #[(1, 101), (5, 78), (4, 37), (2, 15), (10, 62), (7, 99)]



'''
4.12 不同集合上元素的迭代
你想在多个对象执行相同的操作，但是这些对象在不同的容器中，你希望代码在不失可读性的情况下避免写重复的循环。
'''
# itertools.chain() 它接受一个可迭代对象列表作为输入，并返回一个迭代器，有效的屏蔽再多个容器中迭代细节
from itertools import chain
a = [1, 2, 3, 4]
b = ['x', 'y', 'z']
for x in chain(a, b):
    print(x) # 1 2 3 4 x y z

# 使用chain() 的一个常见场景就是当你想对不同的集合中所有元素执行某些操作
# active_items = set()
# inactive_items = set()
#
# for item in chain(active_items, inactive_items):
    # process item

# itertools.chain() 接受一个或多个可迭代对象作为输入参数，然后创建一个迭代器，依次连续的返回
# 每个可迭代对象中的元素。这种方式要比先将序列合并再迭代要高效的多。














