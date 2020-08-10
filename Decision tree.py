#!/usr/bin/env python
# coding: utf-8


import pandas as pd
data=pd.read_csv('iris.csv')
data.head()
y=data[['Species']]
x=data[['Sepal.Length']]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
from sklearn.tree import DecisionTreeClassifier




dct=DecisionTreeClassifier()




dct.fit(x_train,y_train)



y_pred=dct.predict(x_test)



y_test.head()



y_pred[0:5]



from sklearn.metrics import confusion_matrix


confusion_matrix(y_test,y_pred)


(19+8+11)/(19+2+0+4+8+7+1+8+11)


x=data[['Sepal.Width']]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)





dct.fit(x,y)





y_pred=dct.predict(x_test)





y_test.head()





y_pred[0:5]





confusion_matrix(y_test,y_pred)





(15+16+7)/(15+1+4+2+16+6+6+3+7)







