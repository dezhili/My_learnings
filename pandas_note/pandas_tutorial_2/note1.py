import pandas as pd 

# f = open('./ufo.csv', 'r', encoding='utf-8')
'''
1. How do I select a pandas Series from a DataFrame
'''
data = pd.read_csv('./ufo.csv')
# print(data)

print(data.describe())

record_num = data.describe().ix[0, 0]
print(record_num)
print(record_num.dtype)                # int32


print(data.ix[0, :])
print(data.ix[[0,1,2], :])


record1 = data.ix[0, :]
print(record1)
print(record1['Time'])
# for i in range(record_num):
#     record2 = data.ix[i, :]
#     print(record2['Time'])


# print(data['City'])

# data['Location'] = data.City + ',' + data.State
# print(data.head())

# print(data.shape)
# print(data.dtypes)
# print(data.describe(include=['object']))


'''
2. How do I rename columns in a pandas DataFrame
'''
# print(data.columns)
# data.rename(columns={'Colors Reported': 'Colors_Reported', 'Shape Reported':'Shape_Reported'}, inplace=True)
# print(data.columns)
# data_cols = ['city', 'colors reported','shape reported','state','time']
# data.columns = data_cols
# print(data.columns)

# data.columns = data.columns.str.replace(' ','_')
# print(data.columns)


'''
3. How do I remove columns from a pandas DataFrame
'''
print(data.head())
data.drop('Colors Reported', axis=1, inplace=True)
print(data.head())

data.drop(['City','State'], axis=1, inplace=True)
print(data.head())

data.drop([0, 1], axis=0, inplace=True)
print(data.head())
