# -*- coding: utf-8 -*-
import os
import pandas as pd
from pyspark import SparkConf
from pyspark.ml.classification import LogisticRegression, SparkContext
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

os.environ['PYSPARK_PYTHON']='D:\\Anaconda3\\envs\\model\\python.exe'

# a = [x for x in range(5)]
# print(a)
#
# spark = SparkSession \
#     .builder \
#     .master("local[16]")\
#     .appName('my_first_app_name') \
#     .getOrCreate()
# data_1 = pd.read_csv(r'C:\Users\10651\Desktop\model\lucheng_data.csv', encoding='gbk')
# dfData = spark.createDataFrame(data_1)
# c = dfData.rdd.map(
#     lambda row: (Vectors.dense(row[5],row[4]),row[61])
# )
# e = c.map(lambda x: (x[0],x[1])).toDF(["features",'label'])
# f = e.dropna()
# f.show()
# g = c.map(lambda x:(x[0], )).toDF(["features"])
# g.show()
# lr = LogisticRegression(maxIter=10, regParam=0.3)
# lrModel = lr.fit(f)
# print(lrModel.summary)
# pdValue = lrModel.transform(g).select('probability').toPandas()
# d = pdValue.applymap(lambda x :(x[1]))
# fpr, tpr, thresholds = roc_curve(data_1['Y'],d,pos_label = 1)
# plt.plot(fpr,tpr,linewidth=2,label="ROC")
# plt.xlabel("false presitive rate")
# plt.ylabel("true presitive rate")
# plt.ylim(0,1.05)
# plt.xlim(0,1.05)
# plt.legend(loc=4)
# plt.show()
# roc_auc = auc(fpr, tpr)
# gini = 2*roc_auc - 1
# print('AUC:',roc_auc,'\n','GINI:',gini,'\n')

import numpy as np

import matplotlib.pyplot as plt
data_path = "C:\\Users\\10651\\Desktop\\spark-2.3.0-bin-hadoop2.7\\data\\mllib\\sample_linear_regression_data.txt"
sc_conf = SparkConf()
sc_conf.setAppName("app")
sc_conf.setMaster('local[16]')
#sc_conf.set('spark.executor.memory', '2g')
#sc_conf.set('spark.executor.cores', '4')
#sc_conf.set('spark.cores.max', '40')
#sc_conf.set('spark.logConf', True)
sc = SparkContext(conf=sc_conf)
spark = SparkSession(sc)
rawData = sc.textFile(data_path)

def  filterVar(line):
    var = []
    fields = line.split(' ')
    var.append(fields[0])
    var.append(fields[1].split(':')[1])
    var.append(fields[2].split(':')[1])
    var.append(fields[3].split(':')[1])
    var.append(fields[4].split(':')[1])
    var.append(fields[5].split(':')[1])
    var.append(fields[6].split(':')[1])
    var.append(fields[7].split(':')[1])
    var.append(fields[8].split(':')[1])
    var.append(fields[9].split(':')[1])
    var.append(fields[10].split(':')[1])
    return var
Data = rawData.map(lambda line : filterVar(line))
trainingData,testData = Data.randomSplit([0.5,0.5],6)

hasattr(trainingData, "toDF")
## True
traingData = trainingData.toDF().toPandas().values
testData = testData.toDF().toPandas().values
rate = 0.01
#一维变量
a = np.random.normal()  # X1系数
#b = np.random.normal()
c = np.random.normal()
d = np.random.normal()
e = np.random.normal()
xs = np.random.normal()
# 10000 * 10
for i in range(100):  #训练100次
    sum_a = 0
    #sum_b = 0
    sum_c = 0
    sum_d = 0
    sum_e = 0
    for line in traingData:
        sum_a += rate * (float(line[0]) - (a * float(line[1]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) )) * (float(line[1]))
        #sum_b += rate * (float(line[0]) - (a * float(line[1]) + b * float(line[2]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) + xs)) * (float(line[2]))
        sum_c += rate * (float(line[0]) - (a * float(line[1]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) )) * (float(line[3]))
        sum_d += rate * (float(line[0]) - (a * float(line[1]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) )) * (float(line[4]))
        sum_e += rate * (float(line[0]) - (a * float(line[1]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) )) * (float(line[5]))
        #sum_xs+= rate * (float(line[0]) - (a * float(line[1]) + c * float(line[3]) + d * float(line[4]) + e * float(line[5]) + xs)) * (1)
    a += sum_a
    #b += sum_b
    c += sum_c
    d += sum_d
    e += sum_e
    preData = [(a * float(data[1])+c * float(data[3])+d * float(data[4])+e * float(data[5]) )for data in testData]
    index = 0
    sum_error = 0
    for line in testData:
        sum_error += float(preData[index]) - float(line[0])
        index+=1
    print(str(i)+' 次迭代，误差值为：'+str(sum_error))