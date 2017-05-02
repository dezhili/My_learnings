#sklearn model 的属性与功能

from sklearn import datasets
from sklearn.linear_model import LinearRegression


loaded_data = datasets.load_boston()
data_X = loaded_data.data 
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_X,data_y)

# print(model.predict(data_X[:4,:]))
# print(data_y[:4])

print(model.coef_)  #y = 0.1x+0.3
print(model.intercept_)

print(model.get_params())
print(model.score(data_X,data_y))  #R^2 coefficent of determination

#coefficient of determination(决定系数判断回归方程的拟合程度)
# SSE  SSR SST(==SSE+SSR) 
# R^2 = SSR/SST (通过回归方程得出的dependent variable 有number% 能被 
#				independent variable 所解释) 