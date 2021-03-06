# Seq2Bow 模型

## 结构
seq2bow <br/>
|----data/ <br/>
|----logs/ <br/>
|----models/ <br/>
|----src/ <br/>


## data
模型使用数据置于```seq2bow/data/```文件夹下，其文件包括```
answer_id_word, id_docId, id_word, train.txt, valid.txt, word_vector```以及```test```文件夹。

```id_docId ```：无实际使用意义，保证原有模型框架能直接使用，空文件即可；

```train.txt, valid.txt```：训练测试文本，每一行为问答对，使用'##'分割，文本为分词后文本；

```answer_id_word ```: 答案词表，第一列为词序号，第二列为词，使用tab分割，答案词表词频阈值一般较高，之前使用阈值为[5,12]，一般此词表为6000以下效果较好，太大影响效果；

```id_word ```：问句词表，第一列为词序号，第二列为词，使用tab分割，此词表无需设置较高的词频，去掉极低频词即可，尽量能覆盖文本中所有内容；

```word_vector ```：问句词向量，每一行为一个词的词向量，与```id_word ```每行一一对应。

另外```test/```文件夹下放的需要测试的文件，即需要对问句提取隐层特征的问答对。

## src
模型在```src/```文件夹下，运行其中的```main.py```即可训练模型，训练完成后会自动测试```test/```下面的文件，会把生成的隐层文件写在```logs/```文件夹下，与```data/test/```下的文件名对应。

```main.py```一般对应3种模式：```train ```, ```continue ```, ```test ```，分别对应重新训练以及测试，继续训练以及测试，仅仅只有测试。后两个模式需要```models```文件夹下有已经保存过的模型。

## models
模型运行后需要保存的模型保存在该文件夹下

## logs
一些训练以及测试的log文件会保存在该文件夹下，生成的隐层文件也会在该目录下。


