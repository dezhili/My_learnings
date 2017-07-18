'''
1.1 将序列分解为单独的变量
我们有一个包含N个元素的元组或序列，现在想将它分解为N个单独的变量
'''
data = ['ACME', 50, 90.1, (2017,6,23)]
name, shares, price, date = data
print(name, shares, price, date)

# 实际上不仅仅只是元组或列表，只要对象恰好是可迭代的，那么就可以执行分解操作。这包括字符串、文件、迭代器及生成器
s = 'Hello'
a, b, c, d, e = s
_, shares, price, _ = data
print(a, b, c, d, e)


'''
1.2 从任意长度的可迭代对象中分解元素
需要从某个可迭代对象中分解出N个对象，但是这个可迭代对象的长度可能超过N，导致出现"分解的值过多"异常
too many values to unpack

"*表达式"
'''
# def drop_first_last(grades):
#     first, *middle, last = grades
#     return avg(middle)
record = ['Dave', 'dave@example.com', '773-770-771', '847-420-467']
name, email, *phone_numbers = record
print(phone_numbers)

# *修饰的变量也可以位于列表的第一个位置。
*trailing, current = [10, 8, 7, 1, 9, 5, 10, 3]
print(trailing)

# *式的语法在迭代一个变长的元组序列时尤为有用
records = [
    ('foo', 1, 2),
    ('bar', 'hello'),
    ('foo', 3, 4)
]
def do_foo(x, y):
    print('foo', x, y)
def do_bar(s):
    print('bar', s)

for tag, *args in records:
    if tag == 'foo':
        do_foo(*args)
    elif tag == 'bar':
        do_bar(*args)

line = 'nobody:*:-2:-2:Unprivileged User:/var/empty:/usr/bin/false'
uname, *fields, homedir, sh = line.split(':')
print(fields)  #['*', '-2', '-2', 'Unprivileged User']
print(homedir)
print(*fields)

# 在编写执行这类拆分功能的函数时，人们可以假设这是为了实现某种精巧的递归算法。
def sum(items):
    head, *tail = items
    return head + sum(tail) if tail else head
items = [1, 2, 3, 4, 5]
print(sum(items))