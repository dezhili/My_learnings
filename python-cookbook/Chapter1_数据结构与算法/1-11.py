'''
1.11 命名切片
你的程序已经出现一大堆已无法直视的硬编码切片下标，然后想清理下代码
'''
record = '....................100........513.25...........'
cost = int(record[20:23]) * float(record[31:37])
print(cost)

# 命名切片, 避免了大量无法理解的硬编码下标
SHARES = slice(20, 23)
PRICE = slice(31, 37)
cost1 = int(record[SHARES]) * float(record[PRICE])
print(cost1)

# slice() 内置函数创建了一个切片对象，可以被用在任何切片允许使用的地方
items = [0, 1, 2, 3, 4, 5, 6]
a = slice(2, 4)
print(items[a])
print(items[2:4])
del items[a]
print(items)

# 切片对象a  a.start  a.stop  a.step  a.indices(size)->(start, stop, step)
b = slice(5, 50, 2)
print(b.step)
s = 'HelloWorld'
print(b.indices(len(s)))  # (5, 10, 2)

for i in range(*b.indices(len(s))):
    print(s[i])


'''
1.12 序列中出现最多的元素
collections.Counter  most_common()
'''
words = [
    'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
    'the', 'eyes', 'the', 'eyes', 'the', 'eyes'
]
from collections import Counter
word_counts = Counter(words)
print(word_counts)

top_three = word_counts.most_common(3)
print(top_three)

# Counter 对象可以接受任意的由可哈希元素构成的序列对象。在底层实现上，一个Counter对象就是一个字典，
# 将元素映射到它出现的次数上
print(word_counts['eyes'])

morewords = ['why', 'are', 'you', 'not', 'look', 'in', 'my', 'eyes']
for word in morewords:
    word_counts[word] += 1
print(word_counts['eyes'])

# word_counts.update(morewords)
# Counter instance 可以很容易的跟数学运算操作相结合
a = Counter(words)
b = Counter(morewords)
c = a+b
d = a-b
print(c)
print(d)

