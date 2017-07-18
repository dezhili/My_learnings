'''
4.9 排列组合的迭代
你想迭代遍历一个集合中元素的所有可能的排列或组合
'''
# itertools.permutations(集合) -- 产生一个元组序列，每个元组由集合中所有元素的一个可能排列组成
# ('a', 'b', 'c')
# ('a', 'c', 'b')
# ('b', 'a', 'c')
# ('b', 'c', 'a')
# ('c', 'a', 'b')
# ('c', 'b', 'a')
items = ['a', 'b', 'c']
from itertools import permutations
for p in permutations(items):
    print(p)

for p in permutations(items, 2):  # 传递一个可选的长度参数，得到指定长度的所有排列
    print(p)

# itertools.combinations() 得到输入集合中元素的所有的组合
# 在计算组合的时候，一旦元素被选取就会从候选中剔除掉(如果'a'已经被选取了，那么接下来就不会考虑了)
from itertools import combinations
for c in combinations(items, 3):
    print(c)  #('a', 'b', 'c')
for c in combinations(items, 2):
    print(c)

# itertools.combinations_with_replacement() 允许同一个元素被选择多次
from itertools import combinations_with_replacement
for c in combinations_with_replacement(items, 3):
    print(c) #('a', 'a', 'a') ('a', 'a', 'b') ...


'''
4.10 序列上索引值迭代
'''
# enumerate()
my_list = ['a', 'b', 'c']
for idx, val in enumerate(my_list):
    print(idx, val)

# 索引从行号1 开始
for idx, val in enumerate(my_list, 1):
    print(idx, val)
# 这种情况在你遍历文件时想在错误消息中使用行号定位时非常有用
# def parse_data(filename):
#     with open(filename, 'rt') as f:
#         for lineno, line in enumerate(f, 1):
#             fields = line.split()
#             try:
#                 count = int(fields[1])
#                 ...
#             except ValueError as e:
#                 print('Line {}: Parse error: {}'.format(lineno, e))

# enumerate() 对于跟踪某些值在列表出现的位置非常有用。
# 如果你想将一个文件中出现的单词映射到它出现的行号上去，可以很容易的利用enumerate()
from collections import defaultdict
word_summary = defaultdict(list)
with open('myfile.txt', 'r') as f:
    lines = f.readlines()
for idx, line in enumerate(lines):
    words = [w.strip().lower() for w in line.split()]
    for word in words:
        word_summary[word].append(idx)
for idx, val in word_summary.items():
    print(idx, val)
print(word_summary)

# enumerate() 函数返回的是一个 enumerate 对象实例，它是一个迭代器，返回连续的包含一个计数和一个值的元组
# 元组中的值通过在传入序列上调用 next() 返回










