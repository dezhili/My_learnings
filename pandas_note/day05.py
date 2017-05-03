'''
pandas 导入导出数据

pandas可以读取与存取的资料格式有很多种，像csv、excel、json、html与pickle等
'''
import pandas as pd 
import numpy as np 

# data = pd.read_csv('a.csv')
# print(data)

# data.to_pickle('a.pickle')
import pickle

f = open('a.pickle','rb')
a = pickle.load(f)
f.close()
print(a)


