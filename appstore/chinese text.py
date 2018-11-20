from time import time
import jieba
import numpy as np
from keras import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
X_data = []
Y_data = []
# for line in open(r"C:\Users\10651\Desktop\评论数据\975966条总数据\好样本\a.txt","r",encoding='UTF-8'): #设置文件对象并读取每一行文件
#     X_data.append(line.split("||||||")[0])     #将每一行文件加入到list中
#     Y_data.append(line.split("||||||")[1])
# for line in open(r"C:\Users\10651\Desktop\评论数据\975966条总数据\坏样本\b.txt","r",encoding='UTF-8'): #设置文件对象并读取每一行文件
#     X_data.append(line.split("||||||")[0])     #将每一行文件加入到list中
#     Y_data.append(line.split("||||||")[1])
for line in open(r"C:\Users\10651\Desktop\评论数据\975966条总数据\坏样本\c.txt","r",encoding='UTF-8'): #设置文件对象并读取每一行文件
    X_data.append(line.split("||||||")[0])     #将每一行文件加入到list中
    Y_data.append(line.split("||||||")[1])

# X_trainData, X_testData, y_trainData, y_testData = train_test_split(X_data, Y_data, test_size = 0.3)


vectorizer = CountVectorizer(min_df=1e-5) # drop df < 1e-5,去低频词
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(X_data))
words = vectorizer.get_feature_names()

print(words)

train_label = Y_data[:700000]
val_label = Y_data[700000:800000]
test_label = Y_data[800000:]

train_set = tfidf[:700000]
val_set = tfidf[700000:800000]
test_set = tfidf[800000:]
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix

# LogisticRegression classiy model


lr_model = LogisticRegression()
lr_model.fit(train_set, train_label)

print("val mean accuracy: {0}".format(lr_model.score(val_set, val_label)))
y_pred = lr_model.predict(test_set)
print(classification_report(test_label, y_pred))


print(words)
print("how many words: {0}".format(len(words)))
print("tf-idf shape: ({0},{1})".format(tfidf.shape[0], tfidf.shape[1]))
print(tfidf)

#
# encoder = preprocessing.LabelEncoder()
# corpus_encode_label = encoder.fit_transform(corpus_label)

#下面几个是HashingVectorizer， CountVectorizer+TfidfTransformer，TfidfVectorizer， FeatureHasher的正确用法。

#fh = feature_extraction.FeatureHasher(n_features=15,non_negative=True,input_type='string')
#X_train=fh.fit_transform(tokenized_corpus)
#X_test=fh.fit_transform(tokenized_test_corpus)

#fh = feature_extraction.text.HashingVectorizer(n_features=15,non_negative=True,analyzer='word')
#X_train=fh.fit_transform(tokenized_corpus)
#X_test=fh.fit_transform(tokenized_test_corpus)

cv=CountVectorizer(analyzer='word')
transformer=TfidfTransformer()
X_train=transformer.fit_transform(cv.fit_transform(tokenized_train_corpus))
cv2=CountVectorizer(vocabulary=cv.vocabulary_)
transformer=TfidfTransformer()
X_test = transformer.fit_transform(cv2.fit_transform(tokenized_test_corpus))

print(X_train)

#word=cv.get_feature_names()
#weight=X_train.toarray()
#for i in range(len(weight)):
# print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"  
# for j in range(len(word)):  
#            print word[j],weight[i][j] 
# 
# CountVectorizer().fit_transform()
#
# tfidf = TfidfVectorizer(analyzer='word')
# X_train=tfidf.fit_transform(tokenized_train_corpus)
# tfidf = TfidfVectorizer(analyzer='word', vocabulary = tfidf.vocabulary_)
# X_test=tfidf.fit_transform(tokenized_test_corpus)

y_train = y_trainData
y_test = y_testData

def benchmark(clf_class, params, name):
    print("parameters:", params)
    t0 = time()
    clf = clf_class(**params).fit(X_train, y_train)
    print("done in %fs" % (time() - t0))
    if hasattr(clf, 'coef_'):
        print("Percentage of non zeros coef: %f" % (np.mean(clf.coef_ != 0) * 100))
    print("Predicting the outcomes of the testing set")
    t0 = time()
    pred = clf.predict(X_test)
    print("done in %fs" % (time() - t0))
    print("Classification report on test set for classifier:")
    print(clf)
    print()
    print(classification_report(y_test, pred))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix:")
    print(cm)


if __name__ == "__main__":
    print("Testbenching a linear classifier...")
    parameters = {
    'loss': 'hinge',
    'penalty': 'l2',
    'n_iter': 100,
    'alpha': 0.00001,
    'fit_intercept': True,
    }
    benchmark(SGDClassifier, parameters, 'SGD')

# np.random.normal