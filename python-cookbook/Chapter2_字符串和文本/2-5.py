'''
2.5 字符串搜索和替换
你想在字符串中搜索和匹配指定的文本模式
'''
# 对于简单的字面模式，直接使用str.replace()
text = 'yeah, but no, but yeah, but no, but yeah'
print(text.replace('yeah', 'yep'))

# 对于复杂的模式，请使用re模块的sub()
# 11/27/2012 --> 2012-11-27
text = 'Today is 11/27/2012. PyCon starts 3/13/2013.'
import re
print(re.sub(r'(\d+)/(\d+)/(\d+)', r'\3-\1-\2', text))
# \s 指向前面模式的捕获组号


import re
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
print(datepat.sub(r'\3-\1-\2', text))


# 对于更加复杂的替换，可以传递一个替换回调函数来代替
# 一个替换回调函数的参数是一个 match 对象，也就是 match() 或者 find() 返回的对象。 
# 使用 group() 方法来提取特定的匹配部分。回调函数最后返回替换字符串。
from calendar import month_abbr
def change_date(m):
    mon_name = month_abbr[int(m.group(1))]
    return '{} {} {}'.format(m.group(2), mon_name, m.group(3))
print(datepat.sub(change_date, text))


# 除了替换后的结果，如果还想知道多少替换发生了，可以使用re.subn()
newtext, n = datepat.subn(r'\3-\1-\2', text)
print(newtext, '\n', n)



'''
2.6 字符串忽略大小写的搜索替换
你需要以忽略大小写的方式搜索与替换文本字符串
'''
# 为了在文本操作时忽略大小写，需要在使用re模块的时候给这些操作提供re.IGNORECASE标志参数
text = 'UPPER PYTHON, lower python, Mixed Python'
print(re.findall('python', text, flags=re.IGNORECASE))
print(re.sub('python', 'snake', text, flags=re.IGNORECASE))
# UPPER snake, lower snake, Mixed snake
# 根据这个例子，替换字符串并不会自动跟匹配字符串的大小写保持一致。

# 为了修复上面这个问题，可能需要一个辅助函数
def matchcase(word):
    def replace(m):
        text = m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word
    return replace

print(re.sub('python', matchcase('snake'), text, flags=re.IGNORECASE))
