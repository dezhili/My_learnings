'''
本例中使用一个2次函数加上随机的扰动来生成500个点，
然后尝试用1 2 100 次方的多项式对该数据进行拟合。
'''

import matplotlib.pyplot as plt 
import numpy as np
import scipy as sp  
from scipy.stats import norm 
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# 数据生成
x = np.arange(0, 1, 0.002)
y = x ** 2 + norm.rvs(0, size=500, scale=0.1)   # norm 生成随机扰动

# RMSE
def rmse(y_test, y):
    return sp.sqrt(sp.mean((y_test-y)**2))

# R2 (与均值相比的优秀程度) 0表示不如均值， 1表示完美  (clf.score())
def R2(y_test, y_true):
    return 1-  ((y_test-y_true)**2).sum() / ((y_true-y_true.mean())**2).sum()

# R22
def R22(y_test, y_true):
    y_mean = np.array(y_true)
    y_mean[:] = y_mean.mean()
    return 1-rmse(y_test, y_true)/rmse(y_mean, y_true)

plt.scatter(x, y, s=5)
degree = [1, 2, 100]
y_test = []
y_test = np.array(y_test)

for d in degree:
    # clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    # ('linear',LinearRegression(fit_intercept=False))])
    clf = Pipeline([('poly', PolynomialFeatures(degree=d)),
                    ('linear',linear_model.Ridge())]) # 岭回归
    clf.fit(x[:400, np.newaxis], y[:400])
    y_test = clf.predict(x[:, np.newaxis])

    print(clf.named_steps['linear'].coef_)
    print('rmse=%.2f, R2=%.2f, R22=%.2f, clf.score=%.2f' % 
            (rmse(y_test,y), R2(y_test,y), R22(y_test, y), clf.score(x[:, np.newaxis],y)))
    
    plt.plot(x, y_test, linewidth=2)

plt.grid()
plt.legend(['1','2','100'], loc='upper left')
plt.show()    


# 高次多项式过度拟合了训练数据，包括其中大量的噪音，导致其完全丧失了对数据趋势的预测能力。
# 前面也看到，100次多项式拟合出的系数数值无比巨大。
# 人们自然想到通过在拟合过程中限制这些系数数值的大小来避免生成这种畸形的拟合函数。


# 岭（Ridge）回归（使用L2正则化）、Lasso法（使用L1正则化）、
# 弹性网（Elastic net，使用L1+L2正则化）等方法中，都能有效避免过拟合。