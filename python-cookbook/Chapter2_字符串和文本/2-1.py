'''
2.1 使用多个界定符分割字符串
你需要将一个字符串分割为多个字段，但是分隔符(还有周围的空格)并不是固定的。
re.split() 适用于更加灵活的切割字符串
str.split()只适用于非常简单的字符串分割情形，它并不允许有多个分隔符或分隔符周围不确定的空格 
'''
line = 'asdf fjdk; afed, fjek,asdf, foo'
import re
print(re.split(r'[;,\s]\s*', line)) #['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']

# re.split() 特别注意的是 正则表达式是否包含一个括号捕获分组。
# 如果使用了捕获分组，那么被匹配的文本也会出现在结果列表中。
fields = re.split(r'(;|,|\s)\s*', line)
print(fields) #['asdf', ' ', 'fjdk', ';', 'afed', ',', 'fjek', ',', 'asdf', ',', 'foo']

values = fields[::2]
delimiters = fields[1::2] + ['']
print(values)
print(delimiters)
print(''.join(v+d for v,d in zip(values, delimiters))) #asdf fjdk;afed,fjek,asdf,foo

# 如果你不想保留分割字符串到结果列表中去，但仍然需要使用括号来分组正则表达式的话，确保
# 你的分组是非捕获组，形如(?:...)
print(re.split(r'(?:,|;|\s)\s*', line)) # ['asdf', 'fjdk', 'afed', 'fjek', 'asdf', 'foo']



'''
2.2 字符串开头或结尾匹配
你需要通过指定的文本模式去检查字符串的开头或结尾，比如文件名后缀，URL Scheme等
'''
# str.startswith()  str.endswith()
filename = 'spam.txt'
print(filename.endswith('txt'))
url = 'http://www.python.org'
print(url.startswith('http:'))

# 如果想检查多种匹配可能，只需要将所有的匹配项放入到一个元组中去，然后传给startswith() endswith()
import os
filenames = os.listdir('.')
print(filenames)
print([name for name in filenames if name.endswith(('.jpg', '.png'))])
print(any(name.endswith('.py') for name in filenames)) # True

# 下面的这个方法必须要输入一个元组作为参数。
from urllib.request import urlopen
def read_data(name):
    if name.startswith(('http:', 'https:', 'ftp:')):
        return urlopen(name).read()
    else:
        with open(name) as f:
            return f.read()
# 下面的这个方法必须要输入一个元组作为参数。
choices = ['http:', 'ftp:']
url = 'http://www.python.org'
print(url.startswith(tuple(choices)))



# 类似做字符串开头和结尾的检查，也可以使用切边来实现
filename = 'spam.txt'
print(filename[-4:] == '.txt')
url = 'http://www.python.org'
print(url[:5]=='http:' or url[:6]=='htpps:' or url[:4]=='ftp:')


# 用正则表达式实现
import re
print(re.match('http:|https:|ftp:', url)) 

# 检查某个文件夹中是否存在指定的文件类型
# if any(name.endswith(('.c', '.h')) for name in listdir(dirname)):