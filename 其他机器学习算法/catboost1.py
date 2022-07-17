# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 20:15:17 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
from sklearn.preprocessing import Imputer
import catboost as cb
from sklearn.model_selection import train_test_split
import pandas as pd
#读取文件
readFileName="German_credit.xlsx"
#读取excel
df=pd.read_excel(readFileName)
list_columns=list(df.columns[:-1])
x=df.ix[:,:-1]
y=df.ix[:,-1]
names=x.columns
'''
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
imp.fit(data)
x=imp.transform(data)
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

rf=cb.CatBoostRegressor()
rf.fit(x_train, y_train)
print("accuracy on the training subset:{:.3f}".format(rf.score(x_train,y_train)))
print("accuracy on the test subset:{:.3f}".format(rf.score(x_test,y_test)))



