'''
1.15 通过某个字段将记录分组
有一个字典或实例的序列，想根据某个特定的字段比如date 来分组迭代访问
'''
rows = [
    {'address': '5412 N CLARK', 'date': '07/01/2012'},
    {'address': '5148 N CLARK', 'date': '07/04/2012'},
    {'address': '5800 E 58TH', 'date': '07/02/2012'},
    {'address': '2122 N CLARK', 'date': '07/03/2012'},
    {'address': '5645 N RAVENSWOOD', 'date': '07/02/2012'},
    {'address': '1060 W ADDISON', 'date': '07/02/2012'},
    {'address': '4801 N BROADWAY', 'date': '07/01/2012'},
    {'address': '1039 W GRANVILLE', 'date': '07/04/2012'},
]


from operator import itemgetter
from itertools import groupby

rows.sort(key=itemgetter('date'))
# print(rows)

for date,items in groupby(rows, key=itemgetter('date')):
    print(date)
    for i in items:
        print('', i)
# groupby()扫描整个序列并且查找连续相同值。(或根据指定key函数返回值相同)的元素序列。在每次迭代的时候，
# 它会返回一个值和一个迭代器对象，这个迭代器对象可以生成元素值全部等于上面那个值的组中所有对象。



# 如果仅仅想根据date 字段将数据分组到一个大的数据结构中，并允许随机访问，
# 最好使用defaultdict() 构建一个多值字典，
from collections import defaultdict
rows_by_date = defaultdict(list)
for row in rows:
    rows_by_date[row['date']].append(row)

for r in rows_by_date['07/01/2012']:
    print(r)



'''
1.16 过滤序列元素
一个数据序列，想利用一些规则从中提取出需要的值或者缩短序列
'''
# 列表推导式(潜在缺陷 如果输入非常大时会产生非常大的结果集，占用大量内存)
# 如果你对内存非常敏感，可以使用生成器表达式迭代产生过滤的元素
mylist = [1, 4, -5, 10, -7, 2, 3, -1]
print([n for n in mylist if n > 0])

pos = (n for n in mylist if n > 0)
print(pos)
for x in pos:
    print(x)

# 如果过滤规则比较复杂，不能简单地在列表推导或生成器表达式中表达出来。
# 假设过滤的时候需要处理一些异常或其他复杂情况。可以将过滤代码放到一个函数中，然后使用built-in filter()
values = ['1', '2', '-3', '-', '4', 'N/A', '5']
def is_int(val):
    try:
        x = int(val)
        return True
    except ValueError:
        return False
ivals = list(filter(is_int, values))
print(ivals)

# 过滤操作的一个变种是将不符合条件的值用新的值代替。
clip_neg = [n if n>0 else 0 for n in mylist]
print(clip_neg)

# 另一个过滤工具就是 itertoos.compress(),它以一个iterable对象和一个对应的Boolean选择器序列作为输入参数
# 输出iterable对象中对应选择器为True的元素。
addresses = [
    '5412 N CLARK',
    '5148 N CLARK',
    '5800 E 58TH',
    '2122 N CLARK',
    '5645 N RAVENSWOOD',
    '1060 W ADDISON',
    '4801 N BROADWAY',
    '1039 W GRANVILLE',
]
counts = [ 0, 3, 10, 4, 1, 7, 6, 1]
# 将对应count大于5的地址全部输出
from itertools import compress
more5 = [n>5 for n in counts]
print(list(compress(addresses, more5)))