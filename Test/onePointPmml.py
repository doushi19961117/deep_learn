#引入组件
import pandas as pd
import numpy as np
import os
import re
import math
from collections import defaultdict
import sys
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid',{'font.sans-serif':['simhei','Arial']})
#引入组件
sys.path.append(r'C:\Users\10651\Desktop\项目演示内容\code_for_model_platform')
import code_for_model_platform.na_tongzhi_test as na_tongzhi_test
import code_for_model_platform.cor_test as cor_test
import code_for_model_platform.table_x as  table_x
import code_for_model_platform.logit_pic as logit_pic
import code_for_model_platform.auto_bine as auto_bine
import code_for_model_platform.model_build as model_build
import code_for_model_platform.AUC_GINI_KS as AUC_GINI_KS
import code_for_model_platform.test_10 as test_10
from preprocessing.dingshu import Digits_Trans, Strings_Trans, OneThreshold_Trans

#设置路径 读入数据
os.chdir(r'C:\Users\10651\Desktop')
data_1 = pd.read_csv('lucheng_data.csv', encoding='gbk')
data_1
var_list = list(data_1.columns)
var_list.remove('MOBILE_NUMBER')
var_list.remove('CERT_NO')
var_list.remove('NAME')
var_list.remove('INNET_DATE')
var_list.remove('UP_TIME')
var_list.remove('AREA_ID')
var_list.remove('Y')
var_type_dict = defaultdict(list)
#划分类别连续变量
for i in var_list:
    if len(set(data_1[i]))>40 and (data_1[i].dtypes == 'float64' or data_1[i].dtypes == 'int64'):
        var_type_dict['continua'].append(i)
    elif len(set(data_1[i]))<80:
        var_type_dict['discrete'].append(i)
    else:
        var_type_dict['other'].append(i)

#缺失同质检验
#类别变量分析 仅显示存在同质性的变量
discrete_1 = na_tongzhi_test.discrete_variable_table(data_1,var_type_dict['discrete'])
var_tongzhi_list_1 = []
for i in discrete_1:
    if any(discrete_1[i]['Proportion']>=0.9):
        var_tongzhi_list_1.append(i)
#         print(i+'\n',discrete_1[i],'\n')
#类别变量分布情况
for ii in discrete_1:
    vals = list(discrete_1[ii]['Proportion'])
    fig, ax = plt.subplots()#创建子图
    labels = list(discrete_1[ii].index)
    '''
    ax.pie(vals, labels=labels, colors=colors,
      autopct='%1.1f%%', shadow=True, startangle=90,radius=1.2)
    '''
    ax.pie(vals, radius=1,autopct='%1.1f%%',pctdistance=0.75)
    ax.pie([1], radius=0.6,colors='w')
    ax.set(aspect="equal", title=ii)
    #plt.legend()
    plt.legend(labels,bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)
    plt.show()
    print('')
#剔除同质待分析的类别变量
var_discrete_analyse = [x for x in var_type_dict['discrete'] if x not in var_tongzhi_list_1]

#连续变量同质分析
continua_1 = na_tongzhi_test.fenweishu_continuous_variable(data_1,var_type_dict['continua'])
var_tongzhi_list_2 = []
for i in continua_1:
    if (continua_1[i]['estimate'].loc[11]>=0.9) or (continua_1[i]['estimate'].loc[2] == continua_1[i]['estimate'].loc[8]):
        var_tongzhi_list_2.append(i)
#         print(i+'\n',continua_1[i],'\n')
#可视化连续变量层级
for i in var_type_dict['continua']:
    fig, ax = plt.subplots()#创建子图
    sns.boxplot(data = data_1[i],palette=['m'])
    ax.set(title=i)
    plt.show()
#剔除同质待分析的连续变量
var_continua_analyse = [x for x in var_type_dict['continua'] if x not in var_tongzhi_list_2]

#检验连续变量相关系数
var_cor = cor_test.cor_data(data_1,var_continua_analyse)
var_cor_75_dict= {}
#画图展示相关系数热力图
data_cor_h = data_1[var_continua_analyse].corr()  #test_feature => pandas.DataFrame#
mask = np.zeros_like(data_cor_h, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data_cor_h, mask=mask, cmap=cmap,  vmax=1.0, center=0.5,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
#显示存在相关系数大于0.75的变量
for i in var_cor:
    if var_cor[i][1][0]>=0.75:
#         print(i+'\n',var_cor[i][:5],'\n')
        var_cor_75_dict[i]=var_cor[i]

# 在这里由于暂时没有合适聚类算法手动筛选变量
list_remove_1 = []
list_save = []
for j in var_cor_75_dict.keys():
    list_save.append(j)
    for j2 in var_cor_75_dict[j]:
        if j2[0] >= 0.75 and (j2[1] not in list_save):
            list_remove_1.append(j2[1])

# 剔除相关系数后待分析连续变量
var_continua_analyse_2 = [x for x in var_continua_analyse if x not in list_remove_1]

#批量分析类别变量
for k in var_discrete_analyse:
    print(k+'\n',table_x.table_XXX(data_1,k,'Y'),'\n')

#批量分析连续变量
sns.set(style="darkgrid")
for k1 in var_continua_analyse_2:
    print(k1+'\n')
    logit_pic.drawing(data_1,k1,'Y')
    print('\n')

var_continua_analyse_2 = list(set(var_continua_analyse_2).difference(set(['ST_NUM_M1'])))

# 连续变量自动处理
var_continua_for_model = []
var_continua_process = {}
data_1 = pd.read_csv(r'C:\Users\10651\Desktop\lucheng_data.csv', encoding='gbk')
for j1 in var_continua_analyse_2:
    auto_dict = auto_bine.moto_binning(data_1, j1, 'Y')
    data_1[j1 + '_1'] = auto_dict[0]
    var_continua_process[j1] = auto_dict[1]
    var_continua_for_model.append(j1 + '_1')
    var_continua_process[j1]

# 结果显示连续变量自动处理结果
# for i in var_continua_process.items():
#     print(i,'\n')
data_continua = data_1.copy()


#类别变量处理
T_F_data = {True:1,False:0}
data_1['GENDER_1'] = (data_1['GENDER']==1).map(T_F_data)
data_1['PAY_MODE_1'] = (data_1['PAY_MODE'].apply(lambda x:x in [2,5])).map(T_F_data)
data_1['SERVICE_TYPE_1'] = (data_1['SERVICE_TYPE']=='200101AA').map(T_F_data)
data_1['GROUP_FLAG_1'] = (data_1['GROUP_FLAG']==1).map(T_F_data)
data_1['USER_STATUS_1'] = (data_1['USER_STATUS']==11).map(T_F_data)
data_1['FACTORY_DESC_1'] = (data_1['FACTORY_DESC']=='苹果').map(T_F_data)
data_1['DEV_CHANGE_NUM_Y1_1'] = (data_1['DEV_CHANGE_NUM_Y1'].apply(lambda x:x in [4,5,6])).map(T_F_data)
data_1['REAL_HOME_FLAG_M1_1'] = (data_1['REAL_HOME_FLAG_M1']==1).map(T_F_data)
data_1['LIKE_HOME_FLAG_M1_1'] = (data_1['LIKE_HOME_FLAG_M1']==1).map(T_F_data)
data_1['REAL_WORK_FLAG_M1_1'] = (data_1['REAL_WORK_FLAG_M1']==1).map(T_F_data)
data_1['LIKE_WORK_FLAG_M1_1'] = (data_1['LIKE_WORK_FLAG_M1']==1).map(T_F_data)

var_discrete_for_model= ['GENDER_1','PAY_MODE_1','SERVICE_TYPE_1','GROUP_FLAG_1',
                         'USER_STATUS_1','FACTORY_DESC_1','DEV_CHANGE_NUM_Y1_1',
                         'REAL_HOME_FLAG_M1_1','LIKE_HOME_FLAG_M1_1','REAL_WORK_FLAG_M1_1'
                         ,'LIKE_WORK_FLAG_M1_1']
# var_for_model_all = var_discrete_for_model + var_continua_for_model

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelBinarizer, LabelEncoder ,Binarizer
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn_pandas import DataFrameMapper
from pprint import pprint
    #设置路径 读入数据
os.chdir(r'C:\Users\10651\Desktop')
    #指定自定义python库的位置
# from com.dingshu import Digits_Trans, Strings_Trans, OneThreshold_Trans

model_var_final =['FACTORY_DESC_1',
 'daikuan_app_num_m3_1',
 'OUT_CALL_FEE_SECOND_1',
 'GENDER_1',
 'ST_NUM_M1_1',
 'LIKE_HOME_FLAG_M1_1',
 'OUT_FLUX_FEE_SECOND_1',
 'AGE_1',
 'IN_DURA_M1_1',
 'INCR_FEE_FIRST_1',
 'GW_VISIT_CNT_M1_1',
 'GROUP_FLAG_1',
 'DEV_CHANGE_NUM_Y1_1']


def outputPmml(dataTreatedContinua, var_discrete_for_model, var_continua_for_model, included):
    '''
    var_discrete_for_model数据集采用已经处理好的 在piplene中不做记录
    var_continua_for_model数据集中非二元变量自行处理
                                    二元变量在pipline中处理
    :param dataTreatedContinua:
        数据集
    :param var_discrete_for_model:
        类别变量
    :param var_continua_for_model:
        连续变量
    :param included:
        建模变量
    :return:'GENDER_1', 'GROUP_FLAG_1', 'LIKE_HOME_FLAG_M1_1'
    '''
    traingData = dataTreatedContinua
    discrete = ['GENDER','LIKE_HOME_FLAG_M1','GROUP_FLAG','FACTORY_DESC', 'DEV_CHANGE_NUM_Y1','ST_NUM_M1']
#     continua = list(set(included).difference(set(var_discrete_for_model)))
    continua = list(set(included).difference(set(var_discrete_for_model)).difference(set(['ST_NUM_M1_1'])))

    mapper = DataFrameMapper([
         (['GENDER'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['LIKE_HOME_FLAG_M1'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['GROUP_FLAG'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['FACTORY_DESC'],[Strings_Trans(tureStrings = "苹果")]),
         (['DEV_CHANGE_NUM_Y1'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "4.0,5.0,6.0")]),
         (['ST_NUM_M1'],[Imputer(strategy="mean"),OneThreshold_Trans(threshold = 13)]),
        (continua, None)
    ])
    np.set_printoptions(threshold=np.NaN)
    print(mapper.fit_transform(traingData[continua+discrete]))
    pipeline = PMMLPipeline([('mapper', mapper),
                             ("classifier", LogisticRegression(C=1000))])
    pipeline.fit(traingData[continua+discrete],traingData['Y'])
    #user_classpath 指定java的jar包位置
    sklearn2pmml(pipeline, "D:\\LRbin.pmml", user_classpath = ["C:\\Users\\10651\\Desktop\\custom_transformer.jar"])
outputPmml(data_continua, var_discrete_for_model, var_continua_for_model, model_var_final)