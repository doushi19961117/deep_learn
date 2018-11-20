# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 09:56:19 2018

@author: surface
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
#对样本加权重
def sample_weight(Train_t,y,i_0,j_1):
    '''
    y:目标变量名称
    i_0：非目标观测权重
    j_1:目标观测权重
    '''
    
    
    i = 0
    T_0 = Train_t[Train_t[y]==0]
    T_1 = Train_t[Train_t[y]==1]
    Train_0_0 = pd.DataFrame()
    Train_1_1 = pd.DataFrame()
    while i < i_0:
        Train_0_0 = pd.concat([Train_0_0,T_0])
        i += 1
    j = 0
    while j < j_1:
        Train_1_1 = pd.concat([Train_1_1,T_1])
        j += 1
    Train_weight = pd.concat([Train_0_0,Train_1_1])
    return Train_weight


#构造逐步回归筛选变量并建模
def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=False):
    """ 
        X - pandas.DataFrame with candidate features 解释变量 
        y - list-like with the target 目标变量
        initial_list - list of features to start with (column names of X)
        threshold_in - include a feature if its p-value < threshold_in  
        threshold_out - exclude a feature if its p-value > threshold_out
        verbose - whether to print the sequence of inclusions and exclusions
    Returns: list of selected features 
    Always set threshold_in < threshold_out to avoid infinite looping.
    See https://en.wikipedia.org/wiki/Stepwise_regression for the details
    """
    included = initial_list
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # backward step
        model = sm.Logit(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
        
    logit = sm.Logit(y,sm.add_constant(X[included]))
    model_t = logit.fit()
    model_t.summary()        
    return model_t,included  #model_t为模型,included为解释变量



