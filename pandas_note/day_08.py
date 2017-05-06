import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

# plot data

# Series(线性)
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()

# DataFrame
data = pd.DataFrame(np.random.randn(1000,4),index = np.arange(1000),columns=list("ABCD"))
print(data.head())
# print(np.shape(np.arange(10)))
data = data.cumsum()

# plot methods:
# bar , hist, box, kde, area, scatter, hexbin, pie

ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class 1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class 2', ax=ax)
# data.plot()
plt.show()
