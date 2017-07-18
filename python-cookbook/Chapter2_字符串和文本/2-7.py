'''
2.7 最短匹配模式
你正在试着用正则表达式匹配某个文本模式，但是他找到的是模式的最长可能匹配。而你想修改它变成查找最短可能匹配
'''
import re
str_pat = re.compile(r'\"(.*)\"')
text1 = 'Computer says "no."'
print(str_pat.findall(text1))

text2 = 'Computer says "no." Phone says "yes."'
print(str_pat.findall(text2))

# 在正则表达式中*操作符是贪婪的，因此匹配操作会查找最长的可能匹配。
# 比如上面， 返回['no." Phone says "yes.']并不是我们想要的。
# 在模式的*操作符后面加上?修饰符，可以强制匹配算法寻找最短的可能匹配
str_pat = re.compile(r'\"(.*?)\"')
print(str_pat.findall(text2))  #['no.', 'yes.']



'''
2.8 多行匹配模式
你正在试着使用正则表达式去匹配一大块的文本，而你需要跨越多行去匹配
'''
# 很典型的，当你用(.)去匹配任意字符的时候，忘记了点(.)不能匹配换行符的事实
comment = re.compile(r'/\*(.*?)\*/')
text1 = '/* this is a comment */'
text2 = '''/* this is a 
multiline comment */
        '''
print(comment.findall(text1))  #[' this is a comment ']
print(comment.findall(text2))  #[]

#为了修正这个问题，你可以修改模式字符串，增加对换行的支持。
comment = re.compile(r'/\*((?:.|\n)*?)\*/')
print(comment.findall(text2))  #[' this is a \nmultiline comment ']
#在这个模式中，(?:.|\n)指定了一个非捕获组，(也就是它定义了一个仅仅用做匹配，而不能通过单独捕获或编号的组)


#re.DOTALL   -- 可以让正则表达式中的(.)匹配包括换行符在内的任意字符
comment = re.compile(r'/\*(.*?)\*/', re.DOTALL)
print(comment.findall(text2))