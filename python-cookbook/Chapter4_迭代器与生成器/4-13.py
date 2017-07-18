'''
4.13 创建数据处理管道
你想以数据管道(类似Unix管道)的方式迭代处理数据。 比如，你有个大量的数据需要处理，但是不能将它们一次性放入内存中。
'''
# 生成器函数是一个实现管道机制的好办法。假定文件非常大

# 为了处理这些文件，你可以定义一个由多个执行特定任务独立任务的简单生成器函数组成的容器
import os
import fnmatch
import gzip
import bz2
import re

def gen_find(filepat, top):
    '''
    Find all filenames in a directory tree that match a shell wildcard pattern
    '''
    for path, dirlist, filelist in os.walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield os.path.join(path, name)

def gen_opener(filenames):
    '''
    Open a sequence of filenames one at a time producing a file object.
    The file is closed immediately when proceeding to the next iteration.
    '''
    for filename in filenames:
        if filename.endswith('.gz'):
            f = gzip.open(filename, 'rt')
        elif filename.endswith('.bz2'):
            f = bz2.open(filename, 'rt')
        else:
            f = open(filename, 'rt')
        yield f
        f.close()

def gen_concatenate(iterators):
    '''
    Chain a sequence of iterators together into a single sequence
    '''
    for it in iterators:
        yield from it

def gen_grep(pattern, lines):
    '''
    Look for a regex pattern in a sequence of lines
    '''
    pat = re.compile(pattern)
    for line in lines:
        if pat.search(line):
            yield line

# 其实现在可以很容易的将这些函数连起来创建一个处理管道。比如为了查找包含单词python的所有日志行
# lognames = gen_find('access-log*', 'www')
# files = gen_opener(lognames)
# lines = gen_concatenate(files)
# pylines = gen_grep('(?i)python', lines)
# bytecolumn = (line.rsplit(None, 1)[1] for line in pylines)
# bytes = (int(x) for x in bytecolumn if x != '-')
# print('Total', sum(bytes))


# 以管道方式处理数据可以用来解决各类其他问题， 包括解析，读取实时数据，定时轮询
# 重点是要明白 yield 语句作为数据的生产者而 for 循环语句作为数据的消费者。
# 当这些生成器被连在一起后，每个 yield 会将一个单独的数据元素传递给迭代处理管道的下一阶段。
# 在例子最后部分， sum() 函数是最终的程序驱动者，每次从生成器管道中提取出一个元素。

# gen_concatenate() 函数中出现过 yield from 语句，它将 yield 操作代理到父生成器上去。
# 语句 yield from it 简单的返回生成器 it 所产生的所有值。



'''
4.14 展开嵌套的序列
想将一个多层嵌套的序列展开成一个单层列表
'''
# 写一个包含 yield from 语句的递归生成器来轻松解决这个问题
from collections import Iterable
def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            yield from flatten(x)
        else:
            yield x

items = [1, 2, [3, 4, [5, 6], 7], 8]
for x in flatten(items):
    print(x)
# isinstance(x, Iterable) 检查某个元素是否是可迭代的。
# 如果是的话， yield from 就会返回所有子例程的值。最终返回结果就是一个没有嵌套的简单序列了。

# ignore_types and isinstance(x, ignore_types)
# 用来将字符串和字节排除在可迭代对象外，防止将它们再展开成单个的字符。


# yield from 在你想在生成器中调用其他生成器作为子例程的时候非常有用。
# 如果不使用它的话，那么就必须写额外的for 循环。
def flatten(items, ignore_types=(str, bytes)):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, ignore_types):
            for i in flatten(x):
                yield i
        else:
            yield x
            














