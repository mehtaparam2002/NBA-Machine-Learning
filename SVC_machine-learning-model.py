import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

dataset=pd.read_csv("NBA Per Month Data 1996-2015 - Full Year 1996-1997.csv")

x = dataset.drop('POM',axis=1)
dataset = dataset.dropna()

y_labels = dataset['POM']

X_train, X_test, y_train, y_test = train_test_split(x, y_labels, test_size=0.2, random_state=101)


svc = SVC()
svc.fit(X_train, y_train)

# Printing accuracy score & confusion matrix
print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))
print('--------------------------------------')

model = SVC()
scoring = 'accuracy'

kfold = KFold(n_splits = 200, random_state = 7)
cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)

print(cv_results.mean(), cv_results.std())
print(classification_report(y_test, svc.predict(X_test)))

'''cy = svc.predict(x_test)
svc_confm = confusion_matrix(cy, cy_test, [1,0])
sns.heatmap(svc_confm, annot=True, fmt='.2f',xticklabels = ["Admitted", "Rejected"] , yticklabels = ["Admitted", "Rejected"] )
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.title('Support Vector Machine')
plt.show()'''
