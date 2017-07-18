'''
4.1 手动遍历迭代器
你想遍历一个可迭代对象的所有元素，但是却不想使用 for 循环
'''
# 为了手动的遍历可迭代对象，使用next() 并在代码中捕获 StopIteration 异常
# StopIteration 通常用来指示迭代的结尾
# def manual_iter():
#     with open('/etc/passwd') as f:
#         try:
#             while True:
#                 line = next(f)
#                 print(line, end='')
#         except StopIteration:
#             pass

# next()函数 可以通过返回一个指定值来标记结尾，比如 None
# with open('/etc/passwd') as f:
#     while True:
#         line = next(f, None)
#         if line is None:
#             break
#         print(line, end='')


# 在大多数情况下，我们会使用 for 循环语句来遍历一个可迭代对象。
# 但是了解底层迭代机制 很重要， 对迭代做更加精确的控制
items = [1, 2, 3]
it = iter(items)  # Invokes items.__iter__(), Get the iterator
print(type(it))  #<class 'list_iterator'>
print(next(it))  # Invokes it.__next__(), Run the iterator
print(next(it))
print(next(it))
# print(next(it))  # StopIteration


'''
4.2 代理迭代
你构建了一个自定义容器对象，里面包含有列表、元组或其他可迭代对象。 
你想直接在你的这个新容器对象上执行迭代操作。
'''
# 只需要定义一个 __iter__() ， 将迭代操作代理到容器内部对象上
class Node:
    def __init__(self, value):
        self._value = value
        self._children = []

    def __repr__(self):
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        self._children.append(node)

    def __iter__(self):
        return iter(self._children)  #__iter__()将迭代请求传递给内部的_children 属性

if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    for ch in root:
        print(ch)  # Node(1)  Node(2)

'''
Python的迭代器协议需要__iter__() 返回一个实现了__next__()的迭代器对象
这里的iter() 简化了代码，iter(s) 只是简单的通过调用s.__iter__() 返回对应的迭代器对象。
'''








