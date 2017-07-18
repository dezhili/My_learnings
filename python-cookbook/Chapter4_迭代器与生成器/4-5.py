'''
4.5 反向迭代
'''
# reversed()
# 反向迭代仅仅当对象的大小可预先确定或对象实现了__reversed__() 的特殊方法时才生效
a = [1, 2, 3, 4]
for x in reversed(a):
    print(x)

# 如果两者都不符合，必须先将对象转换为一个列表, 消耗大量内存
# f = open('somefile')
# for line in reversed(list(f)):
#     print(line, end='')

# 通过在自定义类上实现 __reversed__() 定义一个反向迭代器，实现反向迭代
class Countdown:
    def __init__(self, start):
        self.start = start

    # Forward iterator
    def __iter__(self):
        n = self.start
        while n > 0:
            yield n
            n -= 1

    # Reverse iterator
    def __reversed__(self):
        n = 1
        while n<= self.start:
            yield n
            n += 1

for rr in reversed(Countdown(30)):
    print(rr)
for rr in Countdown(30):
    print(rr)


'''
4.6 带有外部状态的生成器函数
你想定义一个生成器函数，但是它会调用某个你想暴露给用户使用的外部状态值
'''
# 你想让你的生成器暴露外部状态给用户， 别忘了你可以简单的将它实现为一个类，
# 然后把生成器函数放到 __iter__() 方法中过去。
from collections import deque

class linehistory:
    def __init__(self, lines, histlen=5):
        self.lines = lines
        self.history = deque(maxlen=histlen)

    def __iter__(self):
        for lineno, line in enumerate(self.lines, 1):
            self.history.append((lineno, line))
            yield line

    def clear(self):
        self.history.clear()

# 为了使用这个类，可以将它当做是一个普通的生成器函数。
# 然而，由于可以创建一个实例对象，于是可以访问内部属性值， history clear()
with open('somefile.txt') as f:
    lines = linehistory(f)
    for line in lines:
        if 'python' in line:
            for lineno, hline in lines.history:
                print('{}:{}'.format(lineno, hline), end='')

# 如果在迭代操作时不使用for循环语句，那么得先调用iter() 函数
f = open('somefile.txt')
lines = linehistory(f)
it = iter(lines)
print(next(it))











