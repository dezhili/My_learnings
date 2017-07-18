'''
4.7 迭代器切片
你想得到一个由迭代器生成的切片对象，但是标准切片操作并不能做到。
'''

# itertools.islice() 适用于在迭代器和生成器上做切片操作, 返回一个可以生成指定元素的迭代器
def count(n):
    while True:
        yield n
        n+=1
c = count(0)
import itertools
for x in itertools.islice(c, 10, 20):
    print(x)


'''
4.8 跳过可迭代对象的开头部分
你想遍历一个可迭代对象，但是它开始的某些元素你并不感兴趣，想跳过它们
'''
# itertools.dropwhile(函数对象， 可迭代对象) 返回一个迭代器对象，丢弃原有序列中直到函数返回False之前的所有元素
# 然后返回后面所有元素
with open('somefile.txt') as f:
    for line in f:
        print(line, end='')
# 想跳过开头的注释行
from itertools import dropwhile
with open('somefile.txt') as f:
    for line in dropwhile(lambda line: line.startswith('#'), f):
        print(line, end='')

# 这个例子是基于根据某个测试函数跳过开始的元素，如果已经明确知道了要跳过的元素的个数，
# 可以使用itertools.islice() 代替
from itertools import islice
items = ['a', 'b', 'c', 1, 4, 10, 15]
for x in islice(items, 3, None):
    print(x)

# 跳过一个可迭代对象的开始部分跟通常的过滤不同。
with open('somefile.txt') as f:
    lines = (line for line in f if not line.startswith('#'))
    for line in lines:
        print(line, end='')
# 这样写确实可以跳过开始部分的注释行，但是同样也会跳过文件中其他所有的注释行。








