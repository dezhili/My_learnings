'''
2.17 在字符串中处理html和xml
你想将HTML 或 XML实体如 &entity; 或 &#code; 替换为对应的文本。
再者，你需要转换文本中特定的字符(比如<, >, 或 &)。
'''
# 如果你想替换文本字符串中的 ‘<’ 或者 ‘>’ ，使用 html.escape() 函数可以很容易的完成。
s = 'Elements are written as "<tag>text</tag>".'
import html
print(s)
print(html.escape(s))
print(html.escape(s, quote=False))

#如果你正在处理的是ASCII文本，并且想将非ASCII文本对应的编码实体嵌入进去， 
# 可以给某些I/O函数传递参数 errors='xmlcharrefreplace' 来达到这个目。
s = 'Spicy Jalapeño'
print(s.encode('ascii', errors='xmlcharrefreplace'))


# 有时候，如果你接收到了一些含有编码值的原始文本，需要手动去做替换，通常你只需要使用HTML
# 或XML解析器的一些相关工具函数/方法即可
s = 'Spicy &quot;Jalape&#241;o&quot.'
from html.parser import HTMLParser
p = HTMLParser()
print(p.unescape(s))

t = 'The prompt is &gt;&gt;&gt;'
from xml.sax.saxutils import unescape
print(unescape(t))



'''
2.18 字符串令牌解析
你有一个字符串，想从左至右将其解析为一个令牌流
'''
text = 'foo = 23 + 42 * 10'


