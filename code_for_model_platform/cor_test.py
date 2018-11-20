

# 计算相关系数
from collections import defaultdict
def cor_data(data_1, variable_list):
    '''
    data_1:数据集
    variable_list：需要计算相关系数的列表
    '''
    cor_dataframe = data_1[variable_list].corr() # 计算相关系数
    dict_1 = defaultdict(dict)
    for i in cor_dataframe.index:
        for j in cor_dataframe.columns:
            dict_1[i][j] = cor_dataframe.loc[i][j]

    for i in dict_1.keys():
        for j in dict_1[i].keys():
            dict_1[i][j] = round(float(dict_1[i][j]),3)

    # 对值排降序
    for i in dict_1.keys():
        dict_1[i] = sorted(zip(dict_1[i].values(), dict_1[i].keys()), reverse=True)
    return dict_1


