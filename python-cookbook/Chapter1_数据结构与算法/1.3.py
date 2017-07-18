'''
1.3 保存最后N个元素
保存有限的历史记录可算是collections.deque 的完美应用场景
'''

'''
我们在写查询元素的代码时，通常会使用Yield表达式的生成器函数
'''
# 在多行上面做简单的文本匹配，并返回匹配所在行的最后N行
# deque(maxlen=N)构造函数会新建一个固定大小的队列。当新的元素加入并且这个队列已满的时候，
# 最老的元素会自动被移除掉
from collections import deque

def search(lines, pattern, history=5):
    previous_lines = deque(maxlen=history)
    for line in lines:
        if pattern in line:
            yield line, previous_lines
        previous_lines.append(line)

if __name__ == '__main__':
    with open(r'./somefile.txt') as f:
        for line, prevlines in search(f, 'python', 5):
            for pline in prevlines:
                print(pline, end='')
            print(line, end='')
            print('-'*20)

# deque类可以被用在任何你只需要一个简单队列数据结构的场合。如果你不设置最大队列大小，那么就会得到一个
# 无限大小队列，你可以在队列的两端执行添加和弹出元素的操作。
q = deque()
q.append(1)
q.append(2)
print(q)
q.appendleft(3)
print(q)
q.pop()
print(q)

# 在队列两端插入或删除元素时间复杂度都是O(1), 而在列表的开头插入或删除元素的时间复杂度为O(N)



# 1.4 查找最大或最小的N个元素(heapq) nlargest() nsmallest

import heapq
nums = [1, 8, 2, 23, -7, 56, 18, 2, 23]
print(heapq.nlargest(3, nums))
print(heapq.nsmallest(3, nums))

#nlargest() nsmallest() 都能接受一个关键字参数，用于更复杂的数据结构中
portfolio=[
    {'name':'IBM', 'shares':100, 'price':91.1},
    {'name':'APPLE', 'shares':50, 'price':553.2},
    {'name':'FB', 'shares':200, 'price':21.09},
    {'name':'YHOO', 'shares':45, 'price':16.35},
    {'name':'ACME', 'shares':75, 'price':115.65},
]
cheap = heapq.nsmallest(3, portfolio, key=lambda s: s['price'])
expensive = heapq.nlargest(3, portfolio, key=lambda s: s['price'])
print(cheap)
print(expensive)

# 底层实现里面，首先会将集合数据进行堆排序后放入一个列表中。
heapq.heapify(nums)
print(nums)   #[-7, 1, 2, 2, 8, 56, 18, 23, 23]
# 堆数据结构最重要的特征heap[0]永远是最小的元素
s1 = heapq.heappop(nums)
print(s1)
s2 = heapq.heappop(nums)
print(s2)
s3 = heapq.heappop(nums)
print(s3)

# 如果要查找的元素个数相对比较小的时候，函数nlargest() 和 nsmallest() 很合适。
# 如果N的大小和集合大小接近的时候，通常先排序这个集合然后再切片sorted(items)[:N]
