import pandas as pd
from preprocessing.dingshu import Digits_Trans, OneThreshold_Trans, Strings_Trans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import Imputer, LabelBinarizer
from sklearn2pmml import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn_pandas import DataFrameMapper
import pandas
import os
# path = os.path.dirname(pandas.__path__)

data = pd.read_csv(r'C:\Users\10651\Desktop\lucheng_data.csv', encoding='gbk')
columns = ['GENDER','PAY_MODE','GROUP_FLAG','USER_STATUS',
           'DEV_CHANGE_NUM_Y1','REAL_HOME_FLAG_M1','LIKE_HOME_FLAG_M1',
           'REAL_WORK_FLAG_M1','LIKE_WORK_FLAG_M1',
           'SERVICE_TYPE','FACTORY_DESC','ST_NUM_M1']
# columns = ['GENDER']
mapper = DataFrameMapper([
         (['GENDER'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['LIKE_HOME_FLAG_M1'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['GROUP_FLAG'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "1.0")]),
         (['FACTORY_DESC'],[Strings_Trans(tureStrings = "苹果")]),
         (['DEV_CHANGE_NUM_Y1'],[Imputer(strategy="mean"),Digits_Trans(tureDigits = "4.0,5.0,6.0")]),
         (['ST_NUM_M1'],[Imputer(strategy="mean"),OneThreshold_Trans(threshold = 13,tureSymbol='<=')]),
#          (['ST_NUM_M1_1'],None),
    ])
print(mapper.fit_transform(data[columns].dropna()))
pipeline = PMMLPipeline([('mapper', mapper),
                         ("classifier", LogisticRegression())])
pipeline.fit(data[columns],data['Y'])
sklearn2pmml(pipeline, "D:\\LRbin3.pmml", user_classpath = ["C:\\Users\\10651\\Desktop\\B.jar"])
