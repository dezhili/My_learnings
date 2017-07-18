'''
2.3 用shell通配符匹配字符串
你想使用Unix Shell 中常用的通配符(比如*.py, Dat[0-9]*.csv等)去匹配文本字符串
fnmatch() 函数匹配能力介于简单的字符串方法和强大的正则表达式之间。 
如果在数据处理操作中只需要简单的通配符就能完成的时候，这通常是一个比较合理的方案。

如果你的代码需要做文件名的匹配，最好使用 glob 模块。
'''
# fnmatch -- fnmatch() fnmatchcase()
from fnmatch import fnmatch, fnmatchcase
print(fnmatch('foo.txt', '*.txt'))
print(fnmatch('foo.txt', '?oo.txt'))
print(fnmatch('Dat45.csv', 'Dat[0-9]*'))
names = ['Dat1.csv', 'Dat2.csv', 'config.ini', 'foo.py']
print([name for name in names if fnmatch(name, 'Dat*.csv')])

# fnmatchcase() 完全使用你的模式大小写匹配
print(fnmatchcase('foo.txt', '*.TXT'))


# 这两个函数通常会被忽略的一个特性是在处理非文件名的字符串时候它们也是很有用的。
addresses = [
    '5412 N CLARK ST',
    '1060 W ADDISON ST',
    '1039 W GRANVILLE AVE',
    '2122 N CLARK ST',
    '4802 N BROADWAY',
]
print([addr for addr in addresses if fnmatchcase(addr, '* ST')])
print([addr for addr in addresses if fnmatchcase(addr, '54[0-9][0-9] *CLARK*')])



'''
2.4 字符串匹配和搜索
你相匹配或者搜索特定模式的文本
'''
#如果你想匹配的是字面字符串，那么通常只需要调用基本字符串方法就行，如str.find(), str.startswith()
text = 'yeah, but no, but yeah, but no, but yeah'
print(text.startswith('yeah'))
print(text.find('no'))

#对于复杂的匹配需要使用正则表达式和re模块
text1 = '11/27/2012'
text2 = 'Nov 27, 2012'
import re
if re.match(r'\d+/\d+/\d+', text1):
    print('yes')
else:
    print('no')

#如果想使用同一个模式去做多次匹配，应该先将模式字符串预编译为模式对象
datepat = re.compile(r'\d+/\d+/\d+')
if datepat.match(text1):
    print('yes')
else:
    print('no')

#match()总是从字符串开始去匹配，如果想查找字符串任意部分的模式出现位置，使用findall()
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
print(datepat.findall(text))

#在定义正则表达式的时候，通常会利用括号去捕获分组。后面处理可以分别将每个组的内容提取出来
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
m = datepat.match('11/27/2012')

print(m.group(0))
print(m.group(1))
print(m.groups())

month, day, year = m.groups()
print(month, day, year)

n = datepat.findall(text)
print(n)
for month, day, year in datepat.findall(text):
    print('{}-{}-{}'.format(year, month, day))


# findall()会搜索文本并以列表形式返回所有的匹配。如果想以迭代的方式返回匹配，可以使用finditer()
for m in datepat.finditer(text):
    print(m.groups())