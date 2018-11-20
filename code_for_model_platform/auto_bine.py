# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 12:18:22 2018

@author: surface
"""


import scipy  
from scipy.stats import chisquare  
import numpy as np
import pandas as pd

from scipy.stats import chi2_contingency

import os


#基于卡方分层
def moto_binning(data_0,avg6_shouru,y):
#data_1是数据集,avg6_shouru是需要分类连续变量,y是目标变量
    new_var_dict = {}
    data_1 = data_0.copy()
    xx = [x*2 for x in range(1,50)]

    xx_2 = [np.percentile(data_1['%s'%avg6_shouru].dropna(),x) for x in xx]#按比例算分位点

    T_F_t = {True:1,False:0}
    output_1 = pd.Series()
    for x in set(xx_2):
        T_F = {True:'<=%s'%x,False:'>%s'%x}
        data_1['new_var'] = (data_1['%s'%avg6_shouru]<=x).map(T_F)
        pp = pd.crosstab(data_1['%s'%y],data_1['new_var'])
        dd = np.array([list(pp.iloc[0]),list(pp.iloc[1])])
        pppp = chi2_contingency(dd)
        output_1['<=%s'%x] = pppp[0]
        new_var_dict['<=%s'%x] = (data_1['%s'%avg6_shouru]<=x).map(T_F_t)
    output_1.sort_values(ascending=False,inplace=True)
    try:
        output_1_1 = output_1.iloc[:5]
    except:
        output_1_1 = output_1



    output_2 = pd.Series()
    for i in range(1,45):
        for j in range(i+5,50):
            i_1 = np.percentile(data_1['%s'%avg6_shouru].dropna(),i*2)
            j_1 = np.percentile(data_1['%s'%avg6_shouru].dropna(),j*2)
            T_F_2 = {True:'>%s and <=%s'%(i_1,j_1),False:'other'}
            data_1['new_var_2'] = ((data_1['%s'%avg6_shouru]>i_1) & (data_1['%s'%avg6_shouru]<=j_1)).map(T_F_2)
            pp_2 = pd.crosstab(data_1['%s'%y],data_1['new_var_2'])
            dd_2 = np.array([list(pp_2.iloc[0]),list(pp_2.iloc[1])])
            pppp_2 = chi2_contingency(dd_2)
            output_2['>%s and <=%s'%(i_1,j_1)] = pppp_2[0]
            new_var_dict['>%s and <=%s'%(i_1,j_1)
             ] = ((data_1['%s'%avg6_shouru]>i_1) & (data_1['%s'%avg6_shouru]<=j_1)).map(T_F_t)          
    output_2.sort_values(ascending=False,inplace=True)
    try:
        output_2_1 = output_2.iloc[:5]
    except:
        output_2_1 = output_2
    output_final = pd.concat([output_1_1,output_2_1])
    new_var_final = new_var_dict[output_final.argmax()]
    output_dict={}
    output_dict[0] = new_var_final
    output_dict[1] = output_final.argmax()
    return output_dict

