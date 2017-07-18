'''
2.15 字符串中插入变量
你想创建一个内嵌变量的字符串，变量被他的值所表示的字符串替换掉。
'''
s = '{name} has {n} messages'
print(s.format(name='Guido', n=37))

# 如果要被替换的变量能在变量域中找到，那么就可以结合使用format_map() vars()
name = 'Guido'
n = 37
print(s.format_map(vars()))
# vars() 也适用于对象实例
class Info:
    def __init__(self, name, n):
        self.name = name
        self.n = n
a = Info('Guido', 37)
print(s.format_map(vars(a)))

# format format_map() 不能很好的处理变量缺失的情况
# 避免这种错误的方法就是另外定义一个含有__missing__()方法的字典对象
class safesub(dict):
    """防止key找不到"""
    def __missing__(self, key):
        return '{' + key + '}'
del n
print(s.format_map(safesub(vars())))   #Guido has {n} messages

# 将变量替换步骤用一个工具函数封装起来。
import sys
def sub(text):
    return text.format_map(safesub(sys._getframe(1).f_locals))

name = 'Gudio'
n = 37
print(sub('Hello {name}'))
print(sub('You have {n} messages.'))



'''
2.16 以指定列宽格式化字符串
你有一些长字符串，想以指定的列宽将它们重新格式化
'''
# textwrap
s = "Look into my eyes, look into my eyes, the eyes, the eyes, \
the eyes, not around the eyes, don't look around the eyes, \
look into my eyes, you're under."

import textwrap
print(textwrap.fill(s, 70))
print(textwrap.fill(s, 40))
print(textwrap.fill(s, 40, initial_indent='   '))
print(textwrap.fill(s, 40, subsequent_indent='   '))

