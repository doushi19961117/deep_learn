import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from sklearn.linear_model import SGDClassifier

lr = SGDClassifier(loss='log', penalty = 'l1')
# X_data = []
# Y_data = []
#
# for line in open(r"C:\Users\10651\Desktop\评论数据\200000条总数据\好样本\a.txt","r",encoding='UTF-8'): #设置文件对象并读取每一行文件
#     X_data.append(line)     #将每一行文件加入到list中
#     Y_data.append(1)
# for line in open(r"C:\Users\10651\Desktop\评论数据\200000条总数据\坏样本\b.txt","r",encoding='UTF-8'): #设置文件对象并读取每一行文件
#     X_data.append(line)     #将每一行文件加入到list中
#     Y_data.append(0)


corpus = r'C:\Users\10651\Desktop\评论数据\200000条分词数据\坏样本\b.txt'
sentences = LineSentence(corpus)  # 加载语料,LineSentence用于处理分行分词语料
#sentences1 = word2vec.Text8Corpus(corpus)  #用来处理按文本分词语料
#print('=--=-=-=-=-=',sentences)
model = Word2Vec(sentences, size=12,window=25,min_count=2,workers=5,sg=1,hs=1)  #训练模型就这一句话  去掉出现频率小于2的词
model.save()
# model = Word2Vec(LineSentence(X_data), size=100, window=10, min_count=3,
#             workers=multiprocessing.cpu_count(), sg=1, iter=10, negative=20)


from keras.optimizers import SGD,Adam
from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import statsmodels.api as sm
model_nn = Sequential()
model_nn.add(Dense(2000, input_dim=1000, init = 'uniform'))
model_nn.add(Dropout(0.15))   # 使用Dropout防止过拟合

model_nn.add(Dense(1550, activation='tanh', init = 'uniform'))
model_nn.add(Dropout(0.15))   # 使用Dropout防止过拟合

model_nn.add(Dense(550, activation='tanh', init = 'uniform'))
model_nn.add(Dropout(0.15))   # 使用Dropout防止过拟合

model_nn.add(Dense(1, activation='sigmoid'))

