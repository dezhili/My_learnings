# 2.1 开始使用re
import re
pattern = re.compile(r'hello')  # 将正则表达式编译成Pattern对象
match = pattern.match('hello world')  # 使用Pattern匹配文本，获得匹配结果，无法匹配时将返回None
if match:
    print(match.group()) #使用Match获得分组信息

# Return : hello

# re.compile(strPattern[,flag])
a = re.compile(r"""\d + 
                   \.
                   \d *""", re.X)
b = re.compile(r"\d+\.\d*")

m = re.match(r'hello', 'hello world!')
print(m.group())


# 2.2 re.match
import re
m = re.match(r'(\w+) (\w+)(?P<sign>.*)', 'hello world!')
print('m.string:', m.string)
print('m.re:', m.re)
print('m.pos:', m.pos)
print('m.endpos:', m.endpos)
print('m.lastindex:', m.lastindex)
print('m.lastgroup:', m.lastgroup)

print('m.group(1, 2):', m.group(1, 2))
print('m.groups():', m.groups())
print('m.groupdict():', m.groupdict())
print('m.start(2):', m.start(2))
print('m.end(2):', m.end(2))
print('m.span(2):', m.span(2))
print(r"m.expand(r'\2 \1\3'):", m.expand(r'\2 \1\3'))

 
# 2.3 Pattern
import re
p = re.compile(r'(\w+) (\w+)(?P<sign>.*)', re.DOTALL)
print('p.pattern:', p.pattern)
print('p.flags:', p.flags)
print('p.groups:', p.groups)
print('p.groupindex:', p.groupindex)


# 实例方法
# search()
pattern = re.compile(r'world')
match = pattern.search('hello world!') #search()查找匹配的子串，不存在能匹配的子串时返回None
if match:
    print(match.group())

# split()
p = re.compile(r'\d+')
print(p.split('one1two2three3four4'))

# findall()
print(p.findall('one1two2three3four4'))

# finditer()
for m in p.finditer('one1two2three3four4'):
    print(m.group())

# sub()
q = re.compile(r'(\w+) (\w+)')
s = 'i say, hello world!'
print(q.sub(r'\2 \1', s))
def func(m):
    return m.group(1).title() + ' ' + m.group(2).title()
print(q.sub(func, s))


# subn()
print(q.subn(r'\2 \1', s))
print(q.subn(func, s))




# 小甲鱼正则表达式
print(re.search(r'FishC', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(7, 12), match='FishC'>
print(re.search(r'.', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(0, 1), match='I'>
print(re.search(r'Fish.', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(7, 12), match='FishC'>

print(re.search(r'\.', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(12, 13), match='.'>
print(re.search(r'\d', 'I love 123 FishC.com')) #<_sre.SRE_Match object; span=(7, 8), match='1'>
print(re.search(r'\d\d\d', 'I love 123 FishC.com')) #<_sre.SRE_Match object; span=(7, 10), match='123'>

# 匹配ip地址， 有问题， 范围 192.168.1.1
print(re.search(r'\d\d\d\.\d\d\d\.\d\d\d\.\d\d\d', '192.168.111.123'))
# <_sre.SRE_Match object; span=(0, 15), match='192.168.111.123'>


# []创建一个字符类,只要匹配字符类中的任何一个，都算匹配
print(re.search(r'[aeiou]', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(3, 4), match='o'>
print(re.search(r'[aeiouAEIOU]', 'I love FishC.com!'))
print(re.search(r'[a-z]', 'I love FishC.com!')) #<_sre.SRE_Match object; span=(2, 3), match='l'>
print(re.search(r'[0-9]', 'I love 123 FishC.com!')) #<_sre.SRE_Match object; span=(7, 8), match='1'>

# {} 限定重复匹配的次数
print(re.search(r'ab{3}c', 'abbbc'))  #<_sre.SRE_Match object; span=(0, 5), match='abbbc'>
print(re.search(r'ab{3,10}c', 'abbbbbc'))

# 0-255 之间的数
print(re.search(r'[01]\d\d|2[0-4]\d|25[0-5]', '188')) #<_sre.SRE_Match object; span=(0, 3), match='188'>

# 开始匹配ip地址 ()是个小组 , 这里没有考虑位数 .1 .001
print(re.search(r'(([01]\d\d|2[0-4]\d|25[0-5])\.){3}([01]\d\d|2[0-4]\d|25[0-5])', '192.168.1.1'))
print(re.search(r'(([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])\.){3}([01]{0,1}\d{0,1}\d|2[0-4]\d|25[0-5])', '192.168.1.1'))
# <_sre.SRE_Match object; span=(0, 11), match='192.168.1.1'>






# 元字符 . ^ $ * + ? {} [] \ | ()
print(re.search(r'Fish(C|D)', 'FishD')) #<_sre.SRE_Match object; span=(0, 5), match='FishD'>

# ^ 脱位符 定位开始位置   $定位结束位置
print(re.search(r'^FishC', 'FishC.com'))
print(re.search(r'FishC$', 'I love FishC'))

# \ ()子组  在这里引用对应的子组所匹配的字符串
print(re.search(r'(FishC)\1', 'FishCFishC')) # r'(FishC)\1' == r'FishCFishC'

#[] 字符类 (一般里面的字符都是被当成普通的字符看待) 除了 - \ ^
print(re.search(r'[.]', 'FishC.com')) #<_sre.SRE_Match object; span=(5, 6), match='.'>
print(re.findall(r'[a-z]', 'FishC.com'))
print(re.findall(r'[\n]', 'FishC.com\n'))
print(re.findall(r'[^a-z]', 'FishC.com\n')) #在这里放开头是取反 ['F', 'C', '.', '\n']
print(re.findall(r'[a-z^]', 'FishC.com\n^')) #['i', 's', 'h', 'c', 'o', 'm', '^']

#{} 做重复的事情
print(re.search(r'FishC{3}', 'FishCCCCCCC'))
print(re.search(r'(FishC){1,5}', 'FishCFishCFishC'))

# * == {0,}    + == {1,}   ? == {0,1}

# 贪婪与非贪婪模式  ， 重复匹配默认使用贪婪模式
s = '<html><title>I love FishC.com</title></html>'
print(re.search(r'<.+>', s))
# <..., match='<html><title>I love FishC.com</title></html>'> 启用贪婪模式
# 后面加个? 非贪婪模式
print(re.search(r'<.+?>', s)) #<_sre.SRE_Match object; span=(0, 6), match='<html>'>





# 特殊字符 \d \D \s \S \w \W \b \B ...

# \b 单词边界，这是一个只匹配单词的开始和结尾的零宽断言。  \B匹配非单词边界
print(re.search(r'\bclass\b', 'no class at all')) # <_sre.SRE_Match object; span=(3, 8), match='class'>
print(re.search(r'\bclass\b', 'one subclass is')) # None

# \w 
print(re.findall(r'\w', 'I love FishC.com'))
# ['I', 'l', 'o', 'v', 'e', 'F', 'i', 's', 'h', 'C', 'c', 'o', 'm']



# re.compile() 上面是模块级别的， 下面是编译成模式
# 如果你需要重复的使用某个正则表达式，那么你可以先将该正则表达式编译成模式对象
p = re.compile(r'[A-Z]')
print(type(p))
print(p.search('I love FishC.com!'))
print(p.findall('I love FishC.com!'))

#编译标志, 可以修改正则表达式的工作方式





# 正则表达式的一些使用方法和扩展语法
# search()
result = re.search(r' (\w+) (\w+)', 'I love fishC.com')
print(result) # 返回一个匹配对象, 要使用匹配对象的方法
print(result.group()) #  love fishC
print(result.group(1)) # love
print(result.group(2))
print(result.start())
print(result.span())

# findall() 如果有子组的话，会有不同的情况
# import urllib.request

# def open_url(url):
#     req = urllib.request.Request(url)
#     req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36')
#     page = urllib.request.urlopen(req)
#     html = page.read().decode('utf-8')
#     return html

# def get_img(html):
#     p = r'<img class="BDE_Image" src="([^"]+\.jpg)"'
#     imglist = re.findall(p, html)
#     for each in imglist:
#         print(each)

#     for each in imglist:
#         filename = each.split('/')[-1]
#         urllib.request.urlretrieve(each, filename, None)

# if __name__ == '__main__':
#     url = 'https://tieba.baidu.com/p/3563409202'
#     get_img(open_url(url))


# findall()  获取ip地址时会生成元组列表 --> 非捕获组
# p = r'(([0,1]?\d?\d|2[0-4]\d|25[0-5])\.){3}([0,1]?\d?\d|2[0-4]\d|25[0-5])'
# p = r'(?:(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])\.){3}(?:[0,1]?\d?\d|2[0-4]\d|25[0-5])'









