import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report

dataset = pd.read_csv("NBA Per Month Data 1996-2015 - Full Year 1996-1997.csv")
dataset = dataset.dropna()

x = dataset.drop('+/-',axis=1)
y = dataset['+/-']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)

predictions = lr.predict(x_test)
yes_data = []
no_data = []
for item in predictions:
    if (item > 0):
        yes_data.append(item)
    else:
        no_data.append(item)
    print(item)

print(len(yes_data))
print(len(no_data))

print(classification_report(y_test,predictions))
