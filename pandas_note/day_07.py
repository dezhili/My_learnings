'''
pandas 合并 merge
pandas 中的 merge 和 concat 类似，但主要是用于两组有key column的数矩，
       统一索引的数据，通常也被用在Database中
'''

import pandas as pd 
import numpy as np 

left1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                             'A': ['A0', 'A1', 'A2', 'A3'],
                             'B': ['B0', 'B1', 'B2', 'B3']})
right1 = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                              'C': ['C0', 'C1', 'C2', 'C3'],
                              'D': ['D0', 'D1', 'D2', 'D3']})
print(left1)
print(right1)

#依据key column合并
res1 = pd.merge(left1, right1, on='key')
print(res1)

#合并时 how=['left', 'right', 'outer', 'inner'] 预设值how='inner'
left2 = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                      'key2': ['K0', 'K1', 'K0', 'K1'],
                      'A': ['A0', 'A1', 'A2', 'A3'],
                      'B': ['B0', 'B1', 'B2', 'B3']})
right2 = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                       'key2': ['K0', 'K0', 'K0', 'K0'],
                       'C': ['C0', 'C1', 'C2', 'C3'],
                       'D': ['D0', 'D1', 'D2', 'D3']})
print(left2)
print(right2)

#  依据key1与key2 columns进行合并，
#  并打印出四种结果['left', 'right', 'outer', 'inner']
res2 = pd.merge(left2, right2, on=['key1','key2'], how='inner')
print(res2)
res3 = pd.merge(left2, right2, on=['key1','key2'], how='outer')
print(res3)



#  Indicator=True 会将合并的记录放在新的一列
df1 = pd.DataFrame({'col1':[0,1],
					'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1, 2, 2],
					'col_right':[2, 2, 2]})
print(df1)
print(df2)

res4 = pd.merge(df1, df2, on='col1', how='outer', indicator=True)
print(res4)
res5 = pd.merge(df1, df2, on='col1', how='outer', indicator='indicator_column')
print(res5)




#  依据index合并
left4 = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                     index=['K0', 'K1', 'K2'])
right4 = pd.DataFrame({'C': ['C0', 'C2', 'C3'],
                      'D': ['D0', 'D2', 'D3']},
                     index=['K0', 'K2', 'K3'])
print(left4)
print(right4)

#依据左右资料集的index进行合并，how='outer',并打印出
res6 = pd.merge(left4, right4, left_index=True, right_index=True, how='outer')
print(res6)

#依据左右资料集的index进行合并，how='inner',并打印出
res7 = pd.merge(left4, right4, left_index=True, right_index=True, how='inner')
print(res7)



#解决overlapping(重叠)
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
print(boys)
print(girls)
res8 = pd.merge(boys, girls, on='k', suffixes=['_boy', '_girl'], how='inner')
print(res8)