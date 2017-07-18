'''
2.11 删除字符串中不需要的字符
你想去掉文本字符串开头，结尾或中间不想要的字符，比如空白
'''
# strip()能用于删除开始或结尾的字符。 lstrip() rstrip()分别从左和从右删除操作
# 默认情况下，这些方法会去除空白字符，但是你也可以指定其他字符
# 去除操作不会对字符串的中间文本产生任何影响
s = ' hello world \n'
print(s.strip())
print(s.lstrip())

# Character stripping
t = '-----hello======='
print(t.lstrip('-'))
print(t.strip('-='))

# 去除操作不会对字符串的中间文本产生任何影响
s = ' hello     world \n'
print(s.strip())

# 如果想处理中间的空格，需要使用replace() or regex
print(s.replace(' ', ''))
import re
print(re.sub('\s+', ' ', s))

# 通常你想将字符串strip() 和其他迭代操作相结合，比如从文件中读取多行数据。
# 如果是这样的话，那么生成器表达式就可以大显身手了。
# with open(filename) as f:
#     lines = (line.strip() for line in f)
#     for line in lines:
#         print(line)
#  lines = (line.strip() for line in f)执行数据转换操作。这种方式非常高效，      
#  因为它不需要预先读取所有数据到一个临时列表中去。它仅仅只是创建一个生成器，并且每次返回之前会先执行strip()



'''
2.12 审查清理文本字符串
一些无聊的幼稚黑客在你的网站页面表单中输入文本”pýtĥöñ”，然后你想将这些字符清理掉。
'''
#文本清理问题会涉及到包括文本解析与数据处理等一系列问题。
# str.upper() str.lower() 将文本转为标准格式
# str.replace() re.sub()等简单替换操作能删除或改变指定的字符序列
# unicodedata.normalize() 将unicode文本标准化

#比如，你可能想消除整个区间上的字符或者去除变音符。 
# 为了这样做，你可以使用经常会被忽视的 str.translate() 方法。
# 创建转换表然后使用translate()
s = 'pýtĥöñ\fis\tawesome\r\n'
remap = {
    ord('\t'): ' ',
    ord('\f'): ' ',
    ord('\r'): None
}
a = s.translate(remap)
print(a)

#创建一个更大的表格，让我们删除所有的和音符
#dict.fromkeys() 构造一个字典，每一个Unicode和音符作为键，对应的值为None
# unicodedata.normalize() 将原始输入标准化为分解形式字符
import unicodedata
import sys
cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(chr(c)))
b = unicodedata.normalize('NFD', a)
print(b)
print(b.translate(cmb_chrs))


#作为另一个例子，这里构造一个将所有Unicode数字字符映射到对应的ASCII字符上的表格：
digitmap = { c: ord('0') + unicodedata.digit(chr(c)) 
            for c in range(sys.maxunicode) 
            if unicodedata.category(chr(c)) == 'Nd' }
print(len(digitmap))
x = '\u0661\u0662\u0663'
print(x.translate(digitmap))



# 另一种清理文本的技术涉及到I/O解码与编码函数。
# 这里的思路是先对文本做一些初步的清理， 然后再结合 encode() 或者 decode() 操作来清除或修改它。
b = unicodedata.normalize('NFD', a) # 将原来的文本分解为单独的和音符
print(b.encode('ascii', 'ignore').decode('ascii'))# ASCII编码/解码只是简单的丢弃掉那些字符


