'''
3.11 随机选择
从一个序列中随机抽取若干元素，或者想生成几个随机数。
'''
# random module--大量函数用来产生随机数和随机选择元素。

# random.choice() 从一个序列中随机抽取一个元素
import random
values = [1, 2, 3, 4, 5, 6]
print(random.choice(values))

# random.sample() 提取出N个不同元素的样本用来做进一步的操作
print(random.sample(values, 2))
print(random.sample(values, 3))

# random.shuffle() 仅仅只是想打乱序列中的元素顺序
print(random.shuffle(values))

# random.randint() 生成随机整数
print(random.randint(0, 10))

# random.random() 生成0到1范围内均匀分布的浮点数
print(random.random())



'''
3.12 基本的日期与时间转换
'''
# datetime module  为了表示一个时间段，可以创建一个 timedelta instance
from datetime import timedelta
a = timedelta(days=2, hours=0)
b = timedelta(hours=4.5)
c = a+b
print(c)
print(c.days)
print(c.seconds)
print(c.seconds / 3600)  # 4.5
print(c.total_seconds() / 3600) # 52.5


# 如果你想表示指定的日期与时间，先创建一个 datetime 实例然后使用标准的数学运算
from datetime import datetime
a = datetime(2012, 9, 23)
print(a)  #2012-09-23 00:00:00
print(a + timedelta(days=10))  #2012-10-03 00:00:00

b = datetime(2012, 12, 21)
d = b - a
print(d)  #89 days, 0:00:00
print(d.days)  #89

now = datetime.today()
print(now)  #2017-07-12 08:59:01.634902
print(now + timedelta(minutes=10))  #2017-07-12 09:09:01.634902

# datetime 会自动处理闰年
a = datetime(2012, 3, 1)
b = datetime(2012, 2, 20)
print(a - b)  #10 days, 0:00:00
print((a - b).days)  #10


# 对大多数基本的日期和时间处理问题， datetime 模块以及足够了。
# 如果你需要执行更加复杂的日期操作，比如处理时区，模糊时间范围，节假日计算等等，
# 可以考虑使用 dateutil模块

# dateutil.relativedelta() 但是它会在处理月份(还有它们的天数差距)的时候填充间隙。
a = datetime(2012, 9, 23)
# print(a + timedelta(months=1))  # months is an invalid keyword argument for this function

from dateutil.relativedelta import relativedelta
print(a + relativedelta(months=+1))  #2012-10-23 00:00:00

b = datetime(2012, 12, 21)
d = b - a
print(d)  #89 days, 0:00:00
d = relativedelta(b, a)
print(d)  #relativedelta(months=+2, days=+28)
print(d.months)  #2










