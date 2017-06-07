import tensorflow.contrib.learn as skflow
# import skflow
from sklearn import datasets, metrics, preprocessing
import numpy as np 
import pandas as pd 

df = pd.read_csv('./CHD.csv', header=0)
# print(df.describe())

def my_model(X, y):
    return skflow.models.logistic_regression(X, y)

a = preprocessing.StandardScaler()
X1 = a.fit_transform(df['age'].astype(float))
y1 = df['chd'].values

classifier = TensorFlowEstimator(model_fn=my_model, n_classes=2)
classifier.fit(X1, y1, logdir='./graph')

score = metrics.accuracy_score(df['chd'].astype(float), classifier.predict(X))
print("Accuracy: %f" % score)
