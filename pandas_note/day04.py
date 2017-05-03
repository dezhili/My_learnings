'''
pandas 处理丢失数据
有时候我们导入或处理数据, 会产生一些空的或者是 NaN 数据,
如何删除或者是填补这些 NaN 数据
'''

import pandas as pd 
import numpy as np 

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan

print(df)

print(df.dropna(axis=0 ,how='any'))   #how=[all, any] axis=0表示行
print(df.fillna(value=0))
print(df.isnull())
# print(np.any(df.isnull()==True))
