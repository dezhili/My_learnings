'''
pandas 设置值

我们可以根据自己的需求, 用 pandas 进行更改数据里面的值, 或者加上一些空的,或者有数值的列.

'''
import pandas as pd 
import numpy as np 

dates = pd.date_range('20160101', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)), index=dates, columns=['A','B','C','D'])

df.iloc[2,2] = 1111
df.loc['20160101','B'] = 2222
print(df)

df.A[df.A>4] = 0  #  只有A列中大于4的赋值为0，不是一整行都为0(df.A返回索引)
print(df)
df[df.A>0] = 0  #  大于0的每一行都为0  
print(df)
df['F'] = np.nan
print(df)
df['E'] = pd.Series([1,2,3,4,5,6], index=pd.date_range('20160101',periods=6))
print(df)



