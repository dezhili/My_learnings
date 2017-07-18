'''
2.9 将Unicode 文本标准化
你正在处理Unicode字符串，需要确保所有字符串在底层有相同的表示
'''
s1 = 'Spicy Jalape\u00f1o'
s2 = 'Spicy Jalapen\u0303o'
print(s1)
print(s2)
print(s1 == s2)
print(len(s1))
print(len(s2))

# 如上所示，在需要比较字符串的程序中使用字符的多种表示会产生问题。
# 为了修正这个问题，你可以使用unicodedata模块先将文本标准化
import unicodedata
t1 = unicodedata.normalize('NFC', s1)
t2 = unicodedata.normalize('NFC', s2)
print(t1 == t2)
print(ascii(t1))
t3 = unicodedata.normalize('NFD', s1) # 第一个参数指定字符串标准化的方式
t4 = unicodedata.normalize('NFD', s2)
print(t3 == t4)
print(ascii(t3))

# python同样支持扩展的标准化形式NFKC和NFKD，它们在处理某些字符的时候增加了额外的兼容特性
s = '\ufb01'
print(unicodedata.normalize('NFD', s))
print(unicodedata.normalize('NFKD', s))
print(unicodedata.normalize('NFKC', s))


# 标准化对于任何需要以一直的方式处理Unicode文本的程序都是非常重要的。
# 当处理来自用户输入的字符串而你很难去控制编码的时候尤其如此。
# 在清理和过滤文本的时候字符的标准化也是很重要的。
# combining() 函数可以测试一个字符是否为和音字符
t1 = unicodedata.normalize('NFD', s1)
print(''.join(c for c in t1 if not unicodedata.combining(c)))
# Spicy Jalapeno




'''
2.10 在正则表达式中使用Unicode
你正在使用正则表达式处理文本，但是关注的是Unicode字符处理
'''
#默认情况下，re 模块已经对一些Unicode字符类有了基本的支持。
# \\d已经匹配任意的unicode数字字符
import re
num = re.compile('\d+')
print(num.match('123'))
print(num.match('\u0661\u0662\u0663'))
