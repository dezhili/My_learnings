'''
3.3 数字的格式化输出
你需要将数字格式化后输出，并控制数字的位数、对齐、千位分隔符和其他的细节。
'''
# 格式化输出单个数字的时候，可以使用内置的 format()
x = 1234.56789
print(format(x, '0.2f')) #1234.57
print(format(x, '>10.1f'))  #    1234.6  Right justified in 10 chars
print(format(x, '<10.1f'))  #1234.6      Left
print(format(x, '^10.1f'))  #  1234.6    Center
print(format(x, ','))  #1,234.56789
print(format(x, '0,.1f'))  #1,234.6

# 如果你想使用指数记法，将f改成e或者E(取决于指数输出的大小写形式)。
print(format(x, 'e'))  #1.234568e+03
print(format(x, '0.2E'))  #1.23E+03

# 同时指定宽度和精度的一般形式是 '[<>^]?width[,]?(.digits)?' ，
# 其中 width 和 digits 为整数，？代表可选部分。 同样的格式也被用在字符串的 format() 方法中。
print('The value is {:0,.2f}'.format(x))  #The value is 1,234.57

# 可以使用字符串的translate() 交换千位符
swap_separator = {ord('.'):',', ord(','):'.'}
print(format(x, ',').translate(swap_separator))  #1.234,56789

# 在很多Python代码中会看到使用%来格式化数字的
print('%0.2f'%x)
print('%10.1f'%x)


'''
3.4 二八十六进制整数
你需要转换或者输出使用二进制，八进制或十六进制表示的整数
'''
# bin() oct() hex()
x = 1234
print(bin(x))  #0b10011010010
print(oct(x))  #0o2322
print(hex(x))  #0x4d2

print(format(x, 'b'))  #10011010010
print(format(x, 'o'))  #2322
print(format(x, 'x'))  #4d2

# 整数是有符号的，所以在处理负数的时候，输出结果会包含一个负号
y = -1234
print(format(y, 'b'))  #-10011010010
print(format(y, 'x'))  #-4d2

# 为了以不同的进制转换整数字符串， 简单的使用带有进制的 int()函数
print(int('4d2', 16))  #1234
print(int('10011010010', 2))  #1234

# python 指定八进制的语法不一样。
# 需要确保八进制数的前缀是 0o


























