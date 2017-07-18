'''
1.9 查找两字典的相同点
怎样在两个字典中寻找相同点(比如相同的键 相同的值)
1. 一个字典就是一个键集合与值集合的映射关系。keys()返回一个展现键集合的键视图对象。
    键视图的一个很少被了解的特性是它们也支持集合操作，比如集合并、交、运算。 
    所以，如果你想对集合的键执行一些普通的集合操作，可以直接使用键视图对象而不用先将它们转换成一个set。
2. items()返回一个包含(键，值)对的元素视图对象。这个对象同样也支持集合操作，并且可以被用来查找两字字典
    有哪些相同的键值对
3. values()不支持这里介绍的集合操作。但可以将值集合转换成set，然后再执行集合运算。
'''
a = {
    'x' : 1,
    'y' : 2,
    'z' : 3
}

b = {
    'w' : 10,
    'x' : 11,
    'y' : 2
}
print(type(a.keys()))
print(b.keys())

# 在两字典的keys() items() 返回结果上执行集合操作
print(a.keys() & b.keys())  # Return {'x', 'y'}
print(a.keys() - b.keys())
print(a.items() & b.items())

# 这些操作也可以用于修改或过滤字典元素。比如，假如想以现有字典构造一个排除几个指定键的新字典
# 字典推导
c = {key: a[key] for key in a.keys()-{'z','w'}}
print(c)



'''
1.10 删除序列相同元素并保持顺序
怎样在一个序列上面保持元素顺序的同时消除重复的值
'''
# 如果序列上的值都是hashable类型，那么可以很简单的利用集合或者生成器来解决这个问题。
# 什么是 hashable imutable
# 如果一个对象在自己的生命周期中都有一哈希值(hash value)(hashable)
# 因为这些数据结构内置了哈希值，每个可哈希的对象都内置了__hash__方法，所以可哈希的对象可以通过哈希值进行
# 对比，也可以作为字典的键值和作为set()的参数。
# 所有python中不可改变的对象都是可哈希的，比如字符串 ，元组， 也就是说字典，列表不可哈希

def dedupe1(items):
    seen = set()
    for item in items:
        if item not in seen:
            yield item
        seen.add(item)

a = [1, 5, 2, 1, 9, 1, 5, 10]
# print(list(dedupe(a)))     这个方法仅仅在序列中元素为hashable 的时候才有用。
# 如果想消除元素不可哈希(dict)的序列中重复元素的话，改变一下。

def dedupe2(items, key=None):
    seen = set()
    for item in items:
        val = item if key is None else key(item)
        if val not in seen:
            yield item
            seen.add(val)

# key 参数指定了一个函数，将序列元素转换成 hashable 类型，
b = [{'x':1, 'y':2}, {'x':1, 'y':3}, {'x':1, 'y':2}, {'x':2, 'y':4}]
print(list(dedupe2(b, key=lambda d: (d['x'], d['y']))))
print(list(dedupe2(b, key=lambda d: d['x'])))

c = set(a)
# print(c)    这种方法不能维护元素的顺序

# with open(somefile, 'r') as f:
    # for line in dedupe2(f):
        # ... 
