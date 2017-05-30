import pandas as pd 
import matplotlib.pyplot as  plt 
'''
15. How do I handle missing values in pandas?
'''
# ufo = pd.read_csv('./ufo.csv')
# print(ufo.head())
# print(ufo.describe())
# print(ufo.tail())

# NaN is not a string, rather it's a special value: numpy.nan 
# it stands for 'Not a number ' and indicates a missing value
# print(ufo.isnull().tail())
# print(ufo.notnull().tail())
# print(ufo.isnull().sum())  # the sum method for a DataFrame operates on axis=0
# print(ufo[ufo.City.isnull()].head())
# print(ufo.shape)
# print(ufo.dropna(how='any'))
# print(ufo.dropna(how='all').shape)
# print(ufo.dropna(subset=['City','Shape Reported'], how='any').shape)
# print(ufo['Shape Reported'].value_counts().head()) # value_counts not include missing values by default
# print(ufo['Shape Reported'].value_counts(dropna=False).head())
# ufo['Shape Reported'].fillna(value='VARIOUS', inplace=True)
# print(ufo['Shape Reported'].value_counts().head())


'''
16. What do I need to know about the pandas index?(Part 1)
'''
# drinks = pd.read_csv('./drinks.csv')
# print(drinks.head())
# print(drinks.index)
# print(drinks.columns)
# print(drinks.shape)
# print(drinks[drinks.continent=='South America'])
# print(drinks.loc[23, 'beer_servings'])  # index column
# drinks.set_index('country', inplace=True)
# print(drinks.head())
# print(drinks.index)
# print(drinks.columns)
# print(drinks.shape)
# print(drinks.loc['Brazil', 'beer_servings'])
# drinks.index.name = None
# print(drinks.head())
# drinks.index.name = 'country'
# drinks.reset_index(inplace=True)
# print(drinks.head())
# print(drinks.shape)
# print(drinks.describe())
# print(drinks.describe().loc['25%', 'beer_servings'])

# print(pd.read_table('./u.user', header=None, sep='|').head())



'''
17. What do I need to know about the pandas index?(Part 2)
'''
# print(drinks.continent.head())
# drinks.set_index('country', inplace=True)
# # print(drinks.continent.head())
# # print(drinks.continent.value_counts())
# # print(drinks.continent.value_counts().index)
# # print(drinks.continent.value_counts().values)
# # print(drinks.continent.value_counts()['Africa'])
# # print(drinks.continent.value_counts().sort_values())
# # print(drinks.continent.value_counts().sort_index())
# print(drinks.beer_servings.head())

# people = pd.Series([30000, 85000], index=['Albania','Andorra'], name='population')
# print(people)
# print((drinks.beer_servings * people).head()) # the 2 Series were aligned by indexes
# print(pd.concat([drinks, people], axis=1).head())



'''
18. How do I select multiple rows and columns from a pandas DataFrame?
'''
ufo = pd.read_csv('./ufo.csv')
print(ufo.head(3))
# the loc method is used to select rows and columns by label
# print(ufo.loc[0, :])
# print(ufo.loc[[0, 1, 2], :])
# print(ufo.loc[0:2, :])

# print(ufo.loc[0:2])
# print(ufo.loc[0:2, 'City'])

# print(ufo.loc[0:2, ['City', 'State']])
# print(ufo[['City', 'State']].head(3))

# print(ufo.loc[0:2, 'City':'State'])
# print(ufo.head(3).drop('Time', axis=1))

# print(ufo.loc[ufo.City == 'Oakland', 'State'])
# print(ufo[ufo.City=='Oakland'].State)

# the iloc method is used to select rows and columns by integer position
# print(ufo.iloc[[0, 1], [0, 3]])
# print(ufo.iloc[0:2, 0:4])
# print(ufo.iloc[0:2, :])
# print(ufo[0:2])

# the ix method is used to select rows and columns by label or integer position
drinks = pd.read_csv('./drinks.csv', index_col='country')
print(drinks.head())
print(drinks.ix['Albania', 0])
print(drinks.ix[1, 'beer_servings'])

print(drinks.ix['Albania':'Andorra', 0:2])
print(ufo.ix[0:2, 0:2])
