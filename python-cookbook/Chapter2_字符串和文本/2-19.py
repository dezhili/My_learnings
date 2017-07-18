'''
2.19 实现一个简单的递归下降分析器
暂时不学
'''

'''
2.20 字节字符串上的字符串操作
你想在字节字符串上执行普通的文本操作(比如移除，搜索和替换)
'''
# 字节字符串同样也支持大部分和文本字符串一样的内置操作
data = b'Hello World'
print(data[0:5])
print(data.startswith(b'Hello'))
print(data.split())
print(data.replace(b'Hello', b'Hello Cruel'))

# 这些操作同样也适用于字节数组
data = bytearray(b'Hello World')
print(data[0:5])
print(data.startswith(b'Hello'))
print(data.split())
print(data.replace(b'Hello', b'Hello Cruel'))

# 可以使用正则表达式匹配字节字符串，但是正则表达式必须也是字节串
data = b'FOO:BAR,SPAM'
import re
print(re.split(b'[:,]',data))

# 大多数情况下，在文本字符串上的操作均可用于字节字符串。
# 然而，有些要注意的是。首先，字节字符串的索引操作返回整数而不是单独字符
a = 'Hello World'
print(a[0], a[1])
b = b'Hello World'
print(b[0], b[1])

# 第二点，字节字符串不会提供一个美观的字符串表示，也不能很好的打印出来
# 除非它们先被解码为一个文本字符串
s = b'Hello World'
print(s)
print(s.decode('ascii'))

# 类似的，也不存在适用于字节字符串的格式化操作
# 如果想要格式化字节字符串，得先使用标准的文本字符串，然后将其编码为字节字符串
print('{:10s} {:10d} {:10.2f}'.format('ACME', 100, 490.1).encode('ascii'))