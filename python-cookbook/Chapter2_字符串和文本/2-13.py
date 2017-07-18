'''
2.13 字符串对齐
你想通过某种对齐方式来格式化字符串
'''
# ljust() rjust() center()
text = 'Hello World'
print(text.ljust(20))
print(text.rjust(20))
print(text.center(20))

print(text.rjust(20, '='))
print(text.center(20, '*'))


# format() 使用<,> 或 ^ 字符后面紧跟一个指定的宽度
print(format(text, '>20'))
print(format(text, '<20'))
print(format(text, '^20'))
print(format(text, '=>20s'))
print(format(text, '*^20s'))
print('{:>10s} {:>10s}'.format('Hello', 'World'))
x = 1.2345
print(format(x, '>10'))
print(format(x, '^10.2f'))


'''
2.14 合并拼接字符串
'''
# 如果想要合并的字符串是在一个序列或iterable中， join()
parts = ['Is', 'Chicago', 'Not', 'Chicago?']
print(' '.join(parts))
print(','.join(parts))

a = 'Is Chicago'
b = 'Not Chicago?'
print(a+ ' ' +b)
print('{} {}'.format(a,b))

a = 'Hello' 'World'
print(a)

# 当我们使用+操作符去连接大量字符串的时候是非常低效率的，因为加号连接会引起内存复制以及垃圾回收操作

# 一个比较聪明的技巧是利用生成器表达式转换数据为字符串的同时合并字符串，
data = ['ACME', 50, 91.1]
print(','.join(str(d) for d in data)) #ACME,50,91.1

# 在做连接操作的时候不要多此一举，
a = 'I'
b = 'am'
c = 'boy'
print(a, b, c, sep=':')

# 如果你准备编写构建大量小字符串的输出代码，最好考虑使用生成器函数，利用yield语句产生输出片段
# 原始的生成器函数并不需要知道使用细节，它只负责生成字符串片段
def sample():
    yield 'Is'
    yield 'Chicago'
    yield 'Not'
    yield 'Chicago?'

text = ''.join(sample())
print(text)

# 将字符串片段重定向到I/O
# for part in sample():
#     f.write(part)

def combine(source, maxsize):
    parts = []
    size = 0
    for part in source:
        parts.append(part)
        size += len(part)
        if size > maxsize:
            yield ''.join(parts)
            parts = []
            size = 0
        yield ''.join(parts)

with open('filename', 'w') as f:
    for part in combine(sample(), 32768):
        f.write(part)
