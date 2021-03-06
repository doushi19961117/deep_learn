import pandas as pd
import numpy as np
# 类别变量统计
def discrete_variable_table(data_1, variable_list):
    '''
    data_1:数据集
    variable_list：需要统计的变量名列表
    '''

    discrete_variable_dict = {}
    for k in variable_list:
        kkl = pd.DataFrame(data_1[k].value_counts())  # 创建数据框储存字符变量统计结果
        kkl.rename(columns={k: 'num'}, inplace=True)  # 更改第一列名称计数
        kkl.loc['nan'] = len(data_1[data_1[k].isnull()])  # 添加缺失行统计结果
        kkl['all_data_num'] = len(data_1)  # 添加全部观测数统计结果
        kkl['Proportion'] = kkl['num'] / kkl['all_data_num']  # 添加占比列统计结果
        discrete_variable_dict[k] = kkl
    return discrete_variable_dict


# 数值变量分为图
def fenweishu_continuous_variable(data_1, var_list):
    '''
    data_1:数据集
    var_list：需要统计的变量名列表
    '''
    continuous_variable_dict = {}
    for i in var_list:
        fenw = pd.DataFrame(columns=['percentile', 'estimate'])
        fenw['percentile'] = ['100%max', '99%', '90%', '75% Q3', '50% Median',
                              '25% Q1', '10%', '5%', '1%', '0% Min', 'Nan', 'Nan_rate']
        n_1 = [100, 99, 90, 75, 50, 25, 10, 5, 1, 0]
        x_1 = [np.percentile(data_1[i].dropna(), x) for x in n_1]
        x_1.append(data_1[i].isnull().sum())
        x_1.append(data_1[i].isnull().sum() / len(data_1))
        fenw['estimate'] = x_1
        continuous_variable_dict[i] = fenw
    return continuous_variable_dict