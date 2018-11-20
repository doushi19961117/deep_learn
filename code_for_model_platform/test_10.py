# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 13:27:50 2018

@author: surface
"""
import pandas as pd
import numpy as np

def model_10_split(y_pred,y_true):
    data = pd.DataFrame(y_true.rename('Y',inplace=True)) 
    data['predict'] = y_pred
    def get_stats2(group):
        return{'min':group.min(),'max':group.max(),
               'count':group.count(),'mean':group.mean(),
               'sum':group.sum()}
    data_sorted = data.sort_values('predict', ascending=False)
    data_sorted['rank_1'] = range(len(data_sorted))
    grouping = pd.qcut(data_sorted.rank_1,10,labels=False)
    total_num = dict(data_sorted.Y.groupby(grouping).apply(get_stats2).unstack())['count']
    actual_bad = dict(data_sorted.Y.groupby(grouping).apply(get_stats2).unstack())['sum']
    actual_bad_per = dict(data_sorted.Y.groupby(grouping).apply(get_stats2).unstack())['mean']
    predict_bad = dict(data_sorted.predict.groupby(grouping).apply(get_stats2).unstack())['mean']
    tat = {'total_count':total_num,'actual_bad':actual_bad,'actual_bad_per':actual_bad_per,'predict_bad':predict_bad}
    tat_1 = pd.DataFrame(tat)
    print(tat_1)
    return tat_1



