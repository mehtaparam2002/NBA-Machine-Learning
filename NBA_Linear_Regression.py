import numpy as np
import pandas as pd
​
​
dataset=pd.read_csv("Seasons_Stats.csv")
from sklearn.model_selection import train_test_split
x = dataset.drop('final_funnel_stage',axis=1)
y = dataset['final_funnel_stage']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
​
data_final=pd.read_csv("zeemee_test_binary.csv")
x_finals = data_final.drop('final_funnel_stage',axis=1)
​
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)
​
​
predictions = lr.predict(x_finals)
yes_data = []
no_data = []
for item in predictions:
    if (item == 0):
        yes_data.append(item)
    else:
        no_data.append(item)
    print(item)
​
print(len(yes_data))
print(len(no_data))

print(classification_report(y_test,predictions))
