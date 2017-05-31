# -*- coding: utf-8 -*-
from flask import Flask, Response, json, request
from bottle import run

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词
import yaml
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



app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello World from Data Team.'

@app.route('/sentiment', methods=['GET'])
def sentiment():
    print request.method
    if request.method == 'GET':
        query = request.args.get('query')
        sa, value = predict(unicode(query))

        #return "%s %s %s" % (sa, value, query)
        result = {}
        result['query'] = query
        result['sa'] = sa
        result['value'] = value
        js = json.dumps(result)
        resp = Response(js, status=200, mimetype='application/json')
        return resp

# -------------------------------------------------------------------------------
# 具体计算的函数

def loaddict():
  fr = open(modeldir + '/dict.data')
  dict = pickle.load(fr)
  return dict

datadir = 'data/train'
modeldir = 'model' 

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

print('Loading Dict Data..')
dict = loaddict()

print('loading model......')
with open(modeldir + '/lstm.yml', 'r') as f:    
    yaml_string = yaml.load(f)
model = model_from_yaml(yaml_string)

print('loading weights......')
model.load_weights(modeldir + '/lstm.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



def predict(text):
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
  

  # 至此模型已经载入完成，可以进行预测
  #classes = model.predict_classes(textvec, verbose=1)
  proba = model.predict_proba(textvec, verbose=0)
  for s in proba:
    # 找到最大概率的那个，然后输出对应结果
    if s[0] > s[1] and s[0] > s[2]:
      index = 0
      des = u'neg'
    if s[1] > s[0] and s[1] > s[2]:
      index = 1
      des = u'mid'
    if s[2] > s[0] and s[2] > s[1]:
      index = 2
      des = u'pos'
    #print(des + ' ' + str(s[index]) + ' ' + text)
    return des, str(s[index])


# ------------------------------------------------------------------------------



if __name__ == '__main__':
    run(app=app, reloader=True, port=8777, host='0.0.0.0')