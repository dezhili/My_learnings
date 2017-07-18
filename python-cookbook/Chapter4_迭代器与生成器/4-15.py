'''
4.15 顺序迭代合并后的排序迭代对象
你有一系列排序序列，想将它们合并后得到一个排序序列并在上面迭代遍历。
'''
# heapq.merge()
import heapq
a = [1, 4, 7, 9]
b = [2, 5, 6, 11]
for c in heapq.merge(a, b):
    print(c)  #1 2 4 5 6 7 9 11

# 合并两个排序文件
# with open('sorted_file_1', 'rt') as file1, \
#     open('sorted_file_2', 'rt') as file2, \
#     open('merged_file', 'wt') as outf:
#
#     for line in heapq.merge(file1, file2):
#         outf.write(line)


# 有一点要强调的是 heapq.merge() 需要所有输入序列必须是排过序的。



'''
4.16 迭代器代替while无限循环
你在代码中使用 while 循环来迭代处理数据，因为它需要调用某个函数或者和一般迭代模式不同的测试条件。 
能不能用迭代器来重写这个循环呢？
'''

# CHUNKSIZE = 8192
# def reader(s):
#     while True:
#         data = s.recv(CHUNKSIZE)
#         if data == b'':
#             break
#         process_data(data)
#
# # iter() 代替
# def reader2(s):
#     for chunk in iter(lambda: s.recv(CHUNKSIZE), b''):
#         pass

import sys
f = open('/etc/passwd')
for chunk in iter(lambda: f.read(10), ''):
    n = sys.stdout.write(chunk)