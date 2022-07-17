import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pylab as plt
 
 
#读取文件
readFileName="German_credit.xlsx"
 
#读取excel
df=pd.read_excel(readFileName)
list_columns=list(df.columns[:-1])
x=df.ix[:,:-1]
y=df.ix[:,-1]
names=x.columns
 
train_x, test_x, train_y, test_y=train_test_split(x,y,random_state=0)
 
dtrain=xgb.DMatrix(train_x,label=train_y)
dtest=xgb.DMatrix(test_x)
 
params={'booster':'gbtree',
    #'objective': 'reg:linear',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':4,
    'lambda':10,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':0,
    'nthread':8,
     'silent':1}
 
watchlist = [(dtrain,'train')]
 
bst=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
 
ypred=bst.predict(dtest)
 
# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5)*1
 
#模型校验
from sklearn import metrics
print ('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
print ('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
print ('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
print ('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
print ('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
metrics.confusion_matrix(test_y,y_pred)
 
 
 
print("xgboost:") 
#print("accuracy on the training subset:{:.3f}".format(bst.get_score(train_x,train_y)))
#print("accuracy on the test subset:{:.3f}".format(bst.get_score(test_x,test_y)))
print('Feature importances:{}'.format(bst.get_fscore()))
























