'''
1.13 通过某个关键字排序一个字典列表
根据某个或某几个字典字段来排序这个列表
'''
rows = [
    {'fname':'Brian', 'lname':'Jones', 'uid':1003},
    {'fname':'David', 'lname':'Beazley', 'uid':1002},
    {'fname':'John', 'lname':'Cleese', 'uid':1001},
    {'fname':'Big', 'lname':'Jones', 'uid':1004}    
]

from operator import itemgetter
rows_by_fname = sorted(rows, key=itemgetter('fname'))  #itemgetter()负责创建这个callable对象
rows_by_uid = sorted(rows, key=itemgetter('uid'))
print(rows_by_fname)
print(rows_by_uid)
rows_by_lfname = sorted(rows, key=itemgetter('lname', 'fname'))
print(rows_by_lfname)

# operator.itemgetter() 有一个被rows 中的记录用来查找值的索引参数。
# itemgetter() 有时候可以用 lambda 表达式代替, 但是itemgetter() 运行的快点
rows_by_fname2 = sorted(rows, key=lambda r: r['fname'])
rows_by_lfname = sorted(rows, key=lambda r: (r['lname'], r['fname']))

print(min(rows, key=itemgetter('uid')))
print(max(rows, key=itemgetter('uid')))


'''
1.14 排序不支持原生比较的对象
相排序类型相同的对象，但是他们不支持原生的比较操作
'''
# 内置的sorted() 有一个关键字参数key，可以传入一个callable对象给它，这个callable对象对每个
# 传入的对象返回一个值，这个值会被sorted 用来排序这些对象。
# 比如，如果你在应用程序里面有一个user实例序列，希望通过user_id属性进行排序，可以提供一个
# 以user实例作为输入并输出对应user_id 的callable对象。
class User:
    def __init__(self, user_id):
        self.user_id = user_id
    def __repr__(self):
        return 'User({})'.format(self.user_id)

def sort_notcompare():
    users = [User(23), User(3), User(99)]
    print(users)
    print(sorted(users, key=lambda u: u.user_id))
# sort_notcompare()

# 使用operator.attrgetter() replace lambda function
from operator import attrgetter
print(users, key=attrgetter('user_id'))

# by_name = sorted(users, key=attrgetter9'last_name', 'first_name')
# min(users, key=attrgetter('user_id'))
# max(users, key=attrgetter('user_id'))
