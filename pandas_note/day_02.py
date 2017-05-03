'''
选择数据
方式:
	简单的筛选
	根据标签 loc
	根据序列 iloc
	根据混合的这两种:ix
	通过判断的筛选
'''

import pandas as pd 
import numpy as np 

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['A','B','C','D'])
print(df)
print(df['A'],df.A)  #  选择列
print(df[0:3],df['20160102':'20160104'])  #  选择行
  
# select by label:loc  (纯标签筛选)
print(df.loc['20160102'])
print(df.loc[:,['A','B']])
print(df.loc['20160102',['A','B']])


#  select by position : iloc(通过位置)(纯数字筛选)
print(df.iloc[3])
print(df.iloc[3,1])
print(df.iloc[3:5, 1:3])
print(df.iloc[[1,3,5],1:3])

#  mixed selection : ix  (包含iloc loc)
print(df.ix[:3, ['A','C']])

#  Boolean indexing
print(df)
print(df[df.A > 8])


