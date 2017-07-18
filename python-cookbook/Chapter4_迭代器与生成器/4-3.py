'''
4.3 使用生成器创建新的迭代模式
你想实现一个自定义迭代模式，跟普通的内置函数比如 range() , reversed() 不一样。
'''
# 生成器函数定义 实现一种新的迭代模式。下面是生产某个范围内浮点数的生成器
def frange(start, stop, increment):
    x = start
    while x < stop:
        yield x
        x += increment

# for 循环迭代它 或 使用其他可接收一个可迭代对象的函数 sum() list()
for n in frange(0, 4, 0.5):
    print(n)
print(list(frange(0, 4, 0.5)))  #[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# 一个函数需要有一个 yield 语句即可将其转换为一个生成器。
# 生成器只能用于迭代操作
# 一个生成器函数主要特征是它只会回应在迭代中使用到的 next 操作。 一旦生成器函数返回退出StopIteration，迭代终止。
def countdown(n):
    print('Starting to count from ', n)
    while n > 0:
        yield n
        n -= 1
    print('Done!')
c = countdown(3)  # create the generator

print(next(c))  #3   Run to the first yield and emit a value
print(next(c))  #2
print(next(c))  #1


'''
4.4 实现迭代器协议
构建一个能支持迭代操作的自定义对象，并希望找到一个能实现迭代协议的简单方法
'''
# 生成器函数 Node类表示树形结构， 实现一个以深度优先方式遍历树形节点的生成器
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        yield self
        for c in self:
            yield from c.depth_first()

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))

    for ch in root.depth_first():
        print(ch)
# Node(0) Node(1) Node(3) Node(4) Node(2) Node(5)
# depth_first() 方法简单直观。它首先返回自己本身并迭代每一个子节点并通过调用子节点的depth_first() 方法
# (使用 yield from 语句)返回对应元素。


# Python的迭代协议要求一个 __iter__() 返回一个特殊的迭代器对象，
# 这个迭代器对象实现了__next__() 并通过StopIteration 异常标识迭代的完成。
# 下面我们演示下这种方式，如何使用一个关联迭代器类重新实现 depth_first() 方法：

class Node2:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)

    def depth_first(self):
        return DepthFirstIterator(self)

class DepthFirstIterator(object):
    def __init__(self, start_node):
        self._node = start_node
        self._children_iter = None
        self._child_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        # Return myself if just started; create an iterator for children
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        else:
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)

