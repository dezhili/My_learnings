'''
1.5 实现一个优先级队列
怎样实现一个按优先级排序的队列? 并且在这个队列上面每次pop操作总是返回优先级最高的那个元素
'''
# 利用heapq模块实现一个简单的优先级队列
import heapq

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

class Item:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return 'Item({!r})'.format(self.name)

q = PriorityQueue()
q.push(Item('foo'), 1)
q.push(Item('bar'), 5)
q.push(Item('grok'),1)
print(q.pop())
print(q.pop())
print(q.pop())

# heapq.heappush() heapq.heappop()  分别在队列_queue 上插入和删除第一个元素，并且队列_queue
# 保证第一个元素拥有最高优先级。
# (-priority, index, item)
# 优先级为负数的目的是使得元素按照优先级从高到低排序。 
# 这个跟普通的按优先级从低到高排序的堆排序恰巧相反。
# index 变量的作用是保证同等优先级元素的正确排序。 通过保存一个不断增加的 index 下标变量，
# 可以确保元素按照它们插入的顺序排序。 而且， index 变量也在相同优先级元素比较的时候起到重要作用。



'''
1.6 字典中的键映射多个值
怎样实现一个键对应多个值的字典 multidict?
'''
# 将多个值放到另外的容器中，比如列表或集合里面。
d = {
    'a':[1, 2, 3],
    'b':[4, 5]
}
e = {
    'a':{1, 2, 3},
    'b':{4, 5}
}

# collections.defaultdict --defaultdict的一个特征是它会自动初始化每个key刚开始对应的值，
# 所以只需要关注添加元素操作了。
from collections import defaultdict

# d = defaultdict(list)
# d['a'].append(1)
# d['a'].append(2)
# d['b'].append(4)
# print(d)
# d = defaultdict(set)
# d['a'].add(1)
# d['a'].add(2)
# d['b'].add(4)
# print(d)
# d = {}
# d.setdefault('a', []).append(1)
# d.setdefault('a', []).append(2)
# d.setdefault('b', []).append(4)
# print(d)

# 创建一个多值映射字典
d = defaultdict(list)
for key, value in pairs:
    d[key].append(value)
    