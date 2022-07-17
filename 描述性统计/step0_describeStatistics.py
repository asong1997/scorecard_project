# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 09:40:33 2018

作者邮箱231469242@qq.com
"""

import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from sklearn import metrics
 
#读取文件，去掉-999的单元格
readFileName="German_credit.xlsx"
#读取excel
df=pd.read_excel(readFileName)
#描述性统计
df.describe()
def Variable_distribution(thedf,variable):
    series_variable=thedf[variable]
    distribution=series_variable.value_counts()
    return distribution
 
 
#查看字段缺失率
def Missing_percentage(the_df):
    #字段行总数
    total=df.shape[0]
    #各个字段计数
    count=df.count()
    #空缺率
    miss_percentage=1-count/total*1.0
    return miss_percentage
#缺失率大于百分之50的字段过滤
 
 
#查看字段缺失率
df.info()
miss_percentage=Missing_percentage(df)
#print("missing percentages:",miss_percentage)
#缺失率大于百分之五十的字段过滤
fiter=0.5
columns=['missing_rate']
df_missing=pd.DataFrame(miss_percentage,columns=columns)
df_missing_filter=df_missing[df_missing['missing_rate']>fiter]
print("variables missing rate>%f"%fiter)
print(df_missing_filter)
df_missing.to_excel("missing_variables.xlsx")
df_missing_filter.to_excel("high_missing_variables.xlsx")
 
#变量相关性分析
cor=df.corr('spearman')
#列出高于0.6正相关或小于-0.6负相关的系数矩阵
high_cor=cor[(cor>0.6)|(cor<-0.6)]
 
cor.loc[:,:]=np.tril(cor,k=-1)
cor=cor.stack()
#仅仅列出高相关系数，数据呈现结构化
high_cor1=cor[(cor>0.6)|(cor<-0.6)]
#转换为dataframe结构
df_cor=pd.DataFrame(cor)
df_high_cor1=pd.DataFrame(high_cor1)
#保存到Excel
df_high_cor1.to_excel("high_correlation.xlsx")
df_cor.to_excel("correlation_variables.xlsx")













