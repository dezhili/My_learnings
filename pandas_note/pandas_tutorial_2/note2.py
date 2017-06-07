import pandas as pd 
import matplotlib.pyplot as plt 
'''
4. How do I sort a pandas DataFrame or a Series?
'''
movies = pd.read_csv('./imdb_1000.csv')
# print(movies.head())
# print(movies.title.sort_values().head())
# print(movies.title.sort_values(ascending=False).head())
# print(movies.sort_values('title').head())
# print(movies.sort_values(['content_rating','duration'], ascending=False).head())



'''
5. How do I filter rows of a pandas DataFrame by column value?
'''
print(movies.shape)
# Filter the DataFrame rows to only show movies with a 'duration'of at least 200 minutes
# is_long = movies.duration >= 200
# print(movies[is_long])
# print(movies[movies.duration >= 200])
# print(movies[movies.duration >= 200].genre)
# print(movies.loc[movies.duration >= 200, 'genre'])



'''
6. How do I apply multiple filter criteria to a pandas DataFrame?
'''
# print(movies[(movies.duration >= 200) & (movies.genre=='Drama')])
# print(movies[(movies.duration >= 200) | (movies.genre=='Drama')].head())
# print(movies[(movies.genre=='Crime') | (movies.genre=='Drama') | (movies.genre=='Action')].head(10))
# print(movies[movies.genre.isin(['Crime', 'Drama'])].head(10))



'''
7. How do I read only a subset of the columns or rows when reading from a file
'''
# ufo = pd.read_csv('./ufo.csv', usecols=['City', 'State'])
# print(ufo.columns)

# ufo = pd.read_csv('./ufo.csv', nrows=3)
# print(ufo)


'''
8. How do I iterate through a Series
'''
# for c in ufo.City:
    # print(c)

# iterate through a DataFrame
# for index, row in ufo.iterrows():
#     print(index, row.City, row.State)




'''
9. How do I drop all non_numeric columns from a DataFrame
'''
# drinks = pd.read_csv('./drinks.csv')
# print(drinks.dtypes)
# import numpy as np 
# print(drinks.select_dtypes(include=[np.number]).dtypes)

# how do i know whether I should pass an argument as a string or a list
# print(drinks.describe())
# print(drinks.describe(include='all'))
# print(drinks.describe(include=['object', 'float64']))



'''
10 How do I use the 'axis' parameter in pandas?
'''
# print(drinks.head())
# print(drinks.drop('continent', axis=1).head())
# print(drinks.drop(2, axis=0).head())
# print(drinks.mean())
# print(drinks.mean(axis=0))
# print(drinks.mean(axis=1).head())
# print(drinks.mean(axis='index'))
# print(drinks.mean(axis='columns').head())



'''
11 How do I use string methods in pandas
'''
# imdb = pd.read_csv('./imdb_1000.csv')
# print(imdb.head())
# print(imdb.dtypes)
# print(ufo.country.head())
# print(drinks.country.str.upper().head())
# print(orders.item_name.str.upper().head())
# print(orders.item_name.str.contains('Chicken').head())




'''
12. How do I change the data type of a pandas Series?
'''
# drinks = pd.read_csv('./drinks.csv')
# print(drinks.head())
# print(drinks.dtypes)
# drinks['beer_servings'] = drinks.beer_servings.astype(float)
# print(drinks.dtypes)
# drinks = pd.read_csv('./drinks.csv', dtype={'beer_servings': float})
# print(drinks.dtypes)




'''
13. When should I use a 'groupby' in pandas
'''
# drinks = pd.read_csv('./drinks.csv')
# print(drinks.head())
# print(drinks.beer_servings.mean())
# print(drinks[drinks.continent=='Africa'].beer_servings.mean())
# print(drinks.groupby('continent').beer_servings.mean())
# print(drinks.groupby('continent').beer_servings.max())
# print(drinks.groupby('continent').beer_servings.agg(['count','mean', 'min']))
# print(drinks.groupby('continent').mean())
# drinks.groupby('continent').mean().plot(kind='bar')
# plt.show()




'''
14. How do I explore a pandas Series?
'''
movies = pd.read_csv('./imdb_1000.csv')
print(movies.head())
print(movies.dtypes)
print(movies.genre.describe())
print(movies.genre.value_counts())
print(movies.genre.value_counts(normalize=True))
print(type(movies.genre.value_counts()))
print(movies.genre.value_counts().head())
print(movies.genre.unique())
print(movies.genre.nunique())
print(pd.crosstab(movies.genre, movies.content_rating))
print(movies.duration.describe())
print(movies.duration.mean())
print(movies.duration.value_counts())
movies.duration.plot(kind='hist')
movies.genre.value_counts().plot(kind='bar')
plt.show()




