# -*- coding: utf-8 -*-
from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function

import yaml
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import h5py, pickle, os, datetime
from keras.models import model_from_json, save_model

from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential, model_from_yaml
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras import backend as K
from keras.layers import MaxPooling1D, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
sys.setrecursionlimit(1000000)

# save model http://www.linuxdiyf.com/linux/22940.html
# http://www.linuxdiyf.com/linux/22937.html
#
# http://spaces.ac.cn/archives/3414/
# https://github.com/BUPTLdy/Sentiment-Analysis
# http://blog.sina.com.cn/s/blog_735f29100102wjwu.html
# http://blog.csdn.net/weixin_36541072/article/details/53786020
# https://keras-cn.readthedocs.io/en/latest/getting_started/concepts/
# https://keras-cn.readthedocs.io/en/latest/getting_started/keras_linux/

# LSTM
# http://blog.csdn.net/Dark_Scope/article/details/47056361
# http://blog.csdn.net/zouxy09/article/details/8775360

# 设置参数
# Embedding
maxlen = 128
embedding_size = 128
# Convolution
kernel_size = 5
filters = 64
pool_size = 4
# LSTM
lstm_output_size = 70
lstm_batch_size = 30
lstm_epochs = 15

datadir = 'data/train'
modeldir = 'model/training'
testdir = 'test'

# 加载训练文件
def loadfile():
  print("读取语料数据")
  neg=pd.read_csv(datadir + '/neg.txt', sep='\n',header=None)
  mid=pd.read_csv(datadir + '/mid.txt', sep='\n',header=None) 
  pos=pd.read_csv(datadir + '/pos.txt', sep='\n',header=None)
  
  print("读取训练语料完毕")
  print("给训练语料贴上标签")
  neg['mark']=0
  mid['mark']=1
  pos['mark']=2 
  
  print("合并语料")
  #pn=pd.concat([mid,neg],ignore_index=True)
  pn=pd.concat([pos,mid,neg],ignore_index=True)
  neglen=len(neg)
  midlen=len(mid) #计算语料数目
  poslen=len(pos)

  print('neg count:' + str(neglen))
  print('mid count:' + str(midlen))
  print('pos count:' + str(poslen))
  return pn


def tokenizer(text):
  cw = lambda x: list(jieba.cut(x)) #定义分词函数
  text['words'] = text[0].apply(cw)
  return text

def generatedict(text):
  # 计算词典并保存
  d2v_train = pd.concat([text['words']], ignore_index = True) 
  w = [] #将所有词语整合在一起
  for i in d2v_train:
    w.extend(i)
  dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
  del w,d2v_train
  dict['id'] = list(range(1,len(dict)+1))
  # 这个 dict 需要保存下来
  outputFile = modeldir + '/dict.data'
  fw = open(outputFile, 'w')
  pickle.dump(dict,fw)
  fw.close()
  return dict

def word2index(text, dict):
  get_sent = lambda x: list(dict['id'][x])
  text['sent'] = text['words'].apply(get_sent)
  print("Pad sequences (samples x time)")
  text['sent'] = list(sequence.pad_sequences(text['sent'], maxlen=maxlen))
  return text

def getdata(text):
  X = np.array(list(text['sent'])) #全集
  # 改成三分类需要进行一定的调整，即把 y 转化为向量表示，比如第一类就是 [1,0,0]，第二类就是 [0,1,0]
  Y = np_utils.to_categorical(np.array(list(text['mark'])))

  # 生成训练和测试集
  # random_state 为 1 则表示每次都固定，用于检验，不填或者填 0 为
  x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

  return x_train, y_train, x_test, y_test, X, Y


# CNN + LSTM
# 参数
## Embedding
# maxlen = 50
# embedding_size = 128
## Convolution
# kernel_size = 5
# filters = 64
# pool_size = 4
## LSTM
# lstm_output_size = 70
# lstm_batch_size = 30
# lstm_epochs = 15
# 结果

def cnn_lstm(dict, x, y, xt, yt):
  model = Sequential()
  model.add(Embedding(len(dict)+1, embedding_size, input_length=maxlen))
  model.add(Dropout(0.25))
  model.add(Conv1D(filters, kernel_size, padding='valid',activation='relu',strides=1))
  model.add(MaxPooling1D(pool_size=pool_size))
  model.add(LSTM(lstm_output_size))
  # 这一步用来确定要分多少类，这个 1 表示 1 分类
  model.add(Dense(3)) 
  model.add(Activation('sigmoid'))
  print ('模型构建完成')
  #model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  print ("模型编译完成")
  model.fit(x, y, batch_size=lstm_batch_size, epochs=lstm_epochs, verbose=1)
  print ("模型训练完成")
  print ("保存模型")
  yaml_string = model.to_yaml()
  with open(modeldir + '/lstm.yml', 'w') as outfile:
    outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
  model.save_weights(modeldir + '/lstm.h5')
  print ("测试集评估")
  score = model.evaluate(xt, yt, verbose=0)
  print ("准确率:",score[1])
  return model


# 训练入口函数
def train_lstm(dict,x,y,xt,yt):
  return cnn_lstm(dict,x,y,xt,yt)

def saveresult(model, xt, text):
  classes = model.predict_classes(xt, verbose=1) 
  proba = model.predict_proba(xt, verbose=1)

  print ("\n输出结果")
  filename = 'result.txt'
  f = open('result.txt', 'w')
  i = 1
  j = 0
  for c in classes:
    f.write(str(c))
    f.write(",")
    f.write(str(proba[j]))
    f.write(",")
    line = "".join(text['words'][i])
    f.write(line.encode('utf-8'))
    f.write("\n")
    i = i + 2
    j = j + 1
  f.close()
  print ("\n排序结果")
  num = 1
  result = []
  with open(filename, 'r') as f:
    while True:
      line = f.readline()
      if not line:
        break
      print("processing line #" + str(num))
      num = num + 1
      arr = line.split(',')
      item = (int(arr[0][1:-1]), float(arr[1][2:-1]), "".join(arr[2:]))
      result.append(item)
    result.sort(key=lambda tup:tup[1])
    print(len(result))
    f = open('sorted.txt', 'w')
    for item in result:
      f.write(str(item[0]))
      f.write(",")
      f.write(str(item[1]))
      f.write(",")
      f.write(item[2])
  print("done")

def loaddict():
  fr = open(modeldir + '/dict.data')
  dict = pickle.load(fr)
  return dict


#训练模型，并保存
def train():
  print('Loading Data...')
  pn = loadfile()
  print('Tokenising...')
  pn = tokenizer(pn)
  print('Generating Dict...')
  dict = generatedict(pn)
  print('Word to Index...')
  pn = word2index(pn, dict)
  print('Preparing data...')
  x,y,xt,yt,xa,ya = getdata(pn)
  print('Model Stage...')
  # 这里训练全量模型
  model = train_lstm(dict, xa, ya, xt, yt)
  #print('Save Test Result...')
  #saveresult(model, xt, pn)
  print("Done")
  

def predictarr(filepath):
  dict = loaddict()
  
  with open(modeldir + '/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
  model = model_from_yaml(yaml_string)
  model.load_weights(modeldir + '/lstm.h5')
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
  # 读取测试文件
  # 开始一个 for 循环，外面要统计
  test_count = 0
  if os.path.exists(filepath):
    f = open(filepath, 'r')
    try:
      lines = f.readlines()
      for line in lines:
        if len(line) <= 0:
          continue
        else:
          textarr = list(jieba.cut(line))
          textvec = []
          test_count += 1
          # 这里可能会出现词不存在导致报错的问题，应该是训练数据中没有对应的词，词典有问题
          # TODO 需要把当天的语料加载到词典里，然后再过训练
          for item in textarr:
            # 如果不在词典里，则直接丢弃（因为出现的次数也非常少，不考虑）
            if item in dict['id']:
              textvec.append(dict['id'][item])
          textvec = pd.Series(textvec)  
          textvec = sequence.pad_sequences([textvec], maxlen=maxlen)
          # 概率
          proba = model.predict_proba(textvec, verbose=0)
          for s in proba:
            # 找到最大概率的那个，然后输出对应结果
            index = -1
            des = ''
            if s[0] > s[1] and s[0] > s[2]:
              index = 0
              des = 'neg'
            if s[1] > s[0] and s[1] > s[2]:
              index = 1
              des = 'mid'
            if s[2] > s[0] and s[2] > s[1]:
              index = 2
              des = 'pos'
            print('[' + str(test_count) + ']: ' + des + ' ' + str(s[index]) + ' ' + line[:-1])
    finally:
      f.close() # 确保关闭
  return str(test_count)

def batchtest(filepath):
  dict = loaddict()
  
  with open(modeldir + '/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
  model = model_from_yaml(yaml_string)
  model.load_weights(modeldir + '/lstm.h5')
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  
  # 读取测试文件
  # 开始一个 for 循环，外面要统计
  test_count = 0
  correct_count = 0
  if os.path.exists(filepath):
    f = open(filepath, 'r')
    try:
      lines = f.readlines()
      for line in lines:
        if len(line) <= 0:
          continue
        else:
          arr = line.split(',')
          label = arr[0]
          test_count += 1
          text = ",".join(arr[1:])
          textarr = list(jieba.cut(text))
          textvec = []
          for item in textarr:
            # 如果不在词典里，则直接丢弃（因为出现的次数也非常少，不考虑）
            if item in dict['id']:
              textvec.append(dict['id'][item])
          textvec = pd.Series(textvec)  
          textvec = sequence.pad_sequences([textvec], maxlen=maxlen)
          # 概率
          proba = model.predict_proba(textvec, verbose=0)
          # 判断是否计算正确
          for s in proba:
            if s[0] > 0.5 and label == '1' or s[0] <= 0.5 and label == '0':
              correct_count += 1
              print('[' + str(test_count) + ']: ' + label + ' ' + str(s[0]) + ' ' + text[:-1])
            else:
              print('[' + str(test_count) + ']:[x] ' + label + ' ' + str(s[0]) + ' ' + text[:-1])
    finally:
      f.close() # 确保关闭
  return correct_count, test_count


def predict(text):
  print('Loading Dict Data..')
  dict = loaddict()

  # 把每个词转化为在词典里的数字，更新词典的计数（参考上面的格式）
  textarr = list(jieba.cut(text))
  
  textvec = []
  add = 1
  for item in textarr:
    # 如果不在词典里，则直接丢弃（因为出现的次数也非常少，不考虑）
    if item in dict['id']:
      textvec.append(dict['id'][item])

  textvec = pd.Series(textvec)  
  textvec = sequence.pad_sequences([textvec], maxlen=maxlen)
  
  # ---- 
  print('loading model......')
  with open(modeldir + '/lstm.yml', 'r') as f:
    yaml_string = yaml.load(f)
  model = model_from_yaml(yaml_string)

  print('loading weights......')
  model.load_weights(modeldir + '/lstm.h5')
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # 至此模型已经载入完成，可以进行预测
  #classes = model.predict_classes(textvec, verbose=1)
  proba = model.predict_proba(textvec, verbose=0)
  for s in proba:
    # 找到最大概率的那个，然后输出对应结果
    if s[0] > s[1] and s[0] > s[2]:
      index = 0
      des = 'neg'
    if s[1] > s[0] and s[1] > s[2]:
      index = 1
      des = 'mid'
    if s[2] > s[0] and s[2] > s[1]:
      index = 2
      des = 'pos'
    print(des + ' ' + str(s[index]) + ' ' + text)


# ------------------------------------------

# python train
# python predict sentence
# python test testfile


if __name__=='__main__':
    argvs_length = len(sys.argv)
    if argvs_length >= 2:
        argvs = sys.argv
        action = argvs[1]
        if action == 'train': # 训练
            begin = datetime.datetime.now()
            train()
            end = datetime.datetime.now()
            # 统计训练时间、模型大小，写入到 result.txt 中
            with open(modeldir + '/result.txt', "w") as f:
                f.write('训练时长: ' + str(end-begin))
        elif action == 'predict':
            sentence = " ".join(argvs[2:])
            predict(sentence)
        elif action == 'test':
            testfile = argvs[2]
            begin = datetime.datetime.now()
            result = predictarr(testfile)
            end = datetime.datetime.now()
            print("完成测试，数量", result)
