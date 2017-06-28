# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
import numpy as np
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from scipy import *  
from sys import argv
import jieba, codecs, datetime, time
import sys 
reload(sys) # Python2.5 初始化后会删除 sys.setdefaultencoding 这个方法，我们需要重新载入 
sys.setdefaultencoding('utf-8') 

documents = []
hidden_vectors = []

# 不同业务需要配置不同的词典，效果比较好
# 在停止词中添加的内容，同样需要添加到对应的词典中，这样才能分词出来

user_dict = "dict/zhaohang_dict.txt"
#user_dict = "dict/nanfang_dict.txt"
#user_dict = "dict/pinduoduo_dict.txt"


raw_query = "query/query_5" # 招行
# raw_query = "query/miya.txt" # 密芽 FAQ
# raw_query = "query/clear2" # 有用分期
# raw_query = "query/query_1" # 南方
# hidden_vector = "query/nanfang.other_query.hidden_vec" # 杨老师给的南方的向量，配合 query_1
# raw_query = "query/query_9" # 拼多多

dict_path = "query/zhaohang.dict"
#dict_path = "query/miya.dict"
#dict_path = "query/fenqi.dict"
#dict_path = "query/nanfang.dict"
#dict_path = "query/pinduoduo.dict"

num_topic = 500

topic_dict_path = "query/topic%d.dict"
tokenized_query = "query/tokenized_query"
topic_query = "query/topic_query" # 每个 topic 的 query，用于二次 LDA 聚类
topic_result = "topic_result"
lda_model = "model/lda_model"
lda_model_topic = "model/lda_model_topic"
lsi_model = "model/lsi_model"
stop_dict = "dict/stop_words.txt"

lda_result_dir = "lda_result/"
lsi_result_dir = "lsi_result/"
kmeans_result_dir = "kmeans_result/"
ap_result_dir = "ap_result/"

num_sub_topic = 0
num_topn = 40 # get topic terms 的词个数
final_a = 200 # topic 评分中 doc 的系数
final_b = 100 # topic 评分中 word 的系数
min_word_count = 15 # 分期的数据人名较多，过滤掉
max_sentence_length = 400 # 这个值如果数值大
show_count = 2000 # 处理多少条记录显示一次进度

# 保证每次结果一致
FIXED_SEED = 44 
np.random.seed(FIXED_SEED)

# KL 散度
# 需要传入俩 numpy array
def asymmetricKL(P,Q):  
    return sum(P * log(P / Q)) #calculate the kl divergence between P and Q  

# KL 散度
def symmetricalKL(P,Q):  
    return (asymmetricKL(P,Q)+asymmetricKL(Q,P))/2.00  

def tokenize_query():
    print "开始读取原始用户问句日志，之后分词并保存到文件中"
    count = 0
    jieba.load_userdict(user_dict)
    with open(raw_query) as fr:
        for line in fr:
            if (len(line) < min_word_count):
                # 处理空行和长度小于 6 的
                continue
            if (len(line) > max_sentence_length):
                # 太长的去掉
                continue
            # 去掉空格
            arr = line.split(" ")
            seg_list = jieba.cut("".join(arr))  # 默认是精确模式
            documents.append(" ".join(seg_list))
            count = count + 1
            if (count % 10000 == 0):
                print "已处理 %d 行" % count
    print "日志共 %d 条有效记录" % count
    with codecs.open(tokenized_query, "w+", "utf-8") as f:
        for doc in documents:
            f.write(doc)
    print "写入到文件 %s 完成" % tokenized_query

def load_raw_query():
    print "载入原始数据（不分词不过滤）"
    count = 0
    with open(raw_query) as fr:
        for line in fr:
            documents.append(line)
            count = count + 1
    print "共 %d 条日志记录" % count

def load_hidden_vector():
    print "载入对应 %s 的隐藏向量" % raw_query
    count = 0
    with open(hidden_vector) as fr:
        for line in fr:  
            hidden_vectors.append(line)
            count = count + 1
    print "共 %d 个隐藏向量" % count

def load_tokenized_query():
    print "载入处理后的分词数据"
    with open(tokenized_query) as fr:
        for line in fr:
            documents.append(line)
    print "日志共 %d 条有效记录" % len(documents)

def load_topic_query(topicid):
    print "载入 topic %d 的分词数据" % topicid
    count = 0
    topic_docs = []
    with open("%s%d" % (topic_query, topicid)) as f:
        for line in f:
            topic_docs.append(line)
            count = count + 1
    print "topic %d 共有 %d 条有效记录" % (topicid, len(topic_docs))
    return topic_docs

def get_stoplist():
    stoplist = set()
    # 载入停止词
    with codecs.open(stop_dict, "r+", "utf-8") as f:
        for word in f:
            stoplist.add(word[:-1]) # 去掉最后的回车符号
    print "停止词共 %d 个" % len(stoplist)
    return stoplist

def get_texts():
    stoplist = get_stoplist()
    # remove words and tokenize
    texts = [[word for word in document.lower().split() if word not in stoplist] 
            for document in documents]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    # for i in range(10):
    #     print " ".join(texts[i])
    
    # remove words that appear only once
    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]
    
    # 去掉长度大于 6 的词
    texts = [[token for token in text if len(token) <= 6]
           for text in texts]

    # 删掉空行
    tmp = []
    for text in texts:
        # 至少需要俩词
        if (len(text) > 1):
            tmp.append(text)
    
    texts = tmp

    print "全量语料已载入，共 %d 条" % len(texts)
    return texts

def get_topic_texts(topicid):
    stoplist = get_stoplist()
    topic_docs = load_topic_query(topicid)
    texts = [[unicode(word, "utf-8") for word in doc.lower().split() if unicode(word, "utf-8") not in stoplist] 
            for doc in topic_docs]
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    # remove words that appear only once
    texts = [[token for token in text if frequency[token] > 1]
            for text in texts]
    
    # 去掉长度大于 6 的词
    texts = [[token for token in text if len(token) <= 6]
           for text in texts]

    # 删掉空行
    tmp = []
    for text in texts:
        # 至少需要俩词
        if (len(text) > 1):
            tmp.append(text)
    texts = tmp

    print "topic %d 的语料已载入，共 %d 条" % (topicid, len(texts))
    return texts


def generate_dict():
    texts = get_texts()
    # store the dictionary for future referece
    dictionary = corpora.Dictionary(texts)
    dictionary.save(dict_path) 
    print "词典已保存到 %s 文件中" % dict_path

def generate_topic_dict(topicid):
    texts = get_topic_texts(topicid)
    dictionary = corpora.Dictionary(texts)
    path = topic_dict_path % topicid
    dictionary.save(path)
    print "topic %d 词典已保存到 %s 文件中" % (topicid, path)

def generate_lsi_topic():
     # 载入词典
    dictionary = corpora.Dictionary.load(dict_path)
    print "载入词典完成"

    # 载入语料
    texts = get_texts()

    begin = datetime.datetime.now()
    corpus = [dictionary.doc2bow(text) for text in texts]

    print "开始训练 LSI 模型，共 %d 个 topic" % num_topic
    LSI = models.LsiModel(corpus, num_topics=num_topic, id2word=dictionary, chunksize=2000)

    end = datetime.datetime.now()
    print "处理 LSI 用时", end - begin

    # 保存 LDA 模型
    LSI.save(lsi_model)
    print "模型已保存到 %s 中" % lsi_model

def generate_lda_sub_topic(topicid):
    # 载入词典
    dictionary = corpora.Dictionary.load(topic_dict_path % topicid)
    print "载入 topic %d 词典完成" % topicid

    # 载入语料
    texts = get_topic_texts(topicid)

    begin = datetime.datetime.now()
    corpus = [dictionary.doc2bow(text) for text in texts]
    # store to disk, for later use
    # corpora.MmCorpus.serialize('./nanfang.mm', corpus)  
    # 单核
    # LDA = models.LdaModel(corpus, id2word=dictionary, num_topics=200, update_every=1, minimum_probability=0.1, passes=5)
    # 多核
    # models.ldamulticore.LdaMulticore(corpus, num_topics=200, id2word=dictionary, workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001)
    print "开始训练 topic %d 的子 topic，共 %d 个" % (topicid,  num_sub_topic)
    LDA = models.LdaMulticore(corpus, num_topics=num_sub_topic, id2word=dictionary, workers=4, chunksize=2000, passes=1)
    end = datetime.datetime.now()
    print "训练用时", end - begin

    # 保存 LDA 模型
    path = "%s%d" % (lda_model_topic, topicid)
    LDA.save(path)
    print "topic %d 模型已保存到 %s 中\n" % (topicid, path)

def generate_lda_topic():
    # 载入词典
    dictionary = corpora.Dictionary.load(dict_path)
    print "载入词典完成"

    # 载入语料
    texts = get_texts()

    begin = datetime.datetime.now()
    corpus = [dictionary.doc2bow(text) for text in texts]
    # store to disk, for later use
    # corpora.MmCorpus.serialize('./nanfang.mm', corpus)  
    # 单核
    # LDA = models.LdaModel(corpus, id2word=dictionary, num_topics=200, update_every=1, minimum_probability=0.1, passes=5)
    # 多核
    # models.ldamulticore.LdaMulticore(corpus, num_topics=200, id2word=dictionary, workers=None, chunksize=2000, passes=1, batch=False, alpha='symmetric', eta=None, decay=0.5, offset=1.0, eval_every=10, iterations=50, gamma_threshold=0.001)
    print "开始训练第一层 LDA 模型，共 %d 个 topic" % num_topic
    LDA = models.LdaMulticore(corpus, num_topics=num_topic, id2word=dictionary, workers=4, chunksize=2000, passes=5)
    level1 = datetime.datetime.now()
    print "第一层 LDA 模型训练完成，用时", level1 - begin
    
    # 分离每个 topic 的数据
    # topic_text = [[] for i in range(num_topic)]
    # print "开始分离各个 topic 的数据"
    # for i in range(len(texts)):
    #     if (i % 10000 == 0):
    #         print "正在处理第 %d 行" % i 
    #     # 获取每个文本的 topics
    #     topics = LDA.get_document_topics(corpus[i])
    #     # 这里选择 Top1
    #     if (len(topics) < 1):
    #         continue
    #     # print len(topics), topics[0]
    #     topic_text[topics[0][0]].append(" ".join(texts[i]))    
    
    # 写入每个 topic 的数据
    # for i in range(num_topic):
    #     print "写入 topic %d 的用户问句（已分词）" % i
    #     with codecs.open("%s%d" %(topic_query, i), "w+", "utf-8") as f:
    #         for line in topic_text[i]:
    #             if (len(line) > 1):
    #                 f.write(line+"\n")

    # 写入子 topic
    # for i in range(num_topic):
    #     generate_topic_dict(i) # 生成词典
    #     generate_sub_topic(i) # 生成 子 topic

    end = datetime.datetime.now()
    print "处理 LDA 用时", end - begin

    # 保存 LDA 模型
    LDA.save(lda_model)
    print "模型已保存到 %s 中" % lda_model


def clean_topic(line):
    # 用来清理 topic 的显示
    # example 0.075*"投" + 0.060*"失败" + 0.059*"智定" + 0.043*"邮政" + 0.036*"持仓" + 0.030*"开通" + 0.029*"定" + 0.025*"为什么" + 0.023*"支付" + 0.022*"卡"
    newline = []
    arr = line.split('+')
    for item in arr:
        words = item.split("\"")
        newline.append(words[1])
    return " ".join(newline)

def show_lda_topic_document(rankid, topicid):
    dictionary = corpora.Dictionary.load(topic_dict_path % topicid)
    print "载入 topic %d 词典完成" % topicid

    # 载入语料
    texts = get_topic_texts(topicid)

    corpus = [dictionary.doc2bow(text) for text in texts]
    print "Corpus 数量 %d" % len(corpus)

    # 载入模型
    LDA = models.LdaModel.load("%s%d" % (lda_model_topic, topicid))

    # 创建不同的数组，根据 topic 来排序
    topic_result = [{} for i in range(num_sub_topic)]
    for i in range(len(texts)):
        if (i % 1000 == 0):
            print "正在处理第 %d 行" % i 
        # 获取每个文本的 topics
        topics = LDA.get_document_topics(corpus[i])
        # 这里选择 Top1
        if (len(topics) < 1):
            continue
        # print len(topics), topics[0]
        topic_result[topics[0][0]]["".join(texts[i])] = topics[0][1]

    # 对每个 topic 进行打分
    topic_rank = []
    for i in range(num_sub_topic):
        if (len(topic_result[i]) < 1):
            continue

        temp_doc_vector = []
        # doc score 背景噪声向量
        doc_bg_vector = np.array([1.0 / len(topic_result[i])] * len(topic_result[i]))
        # 得出 doc score 向量
        for record in topic_result[i].items():
            temp_doc_vector.append(record[1]) # 加入各个概率
        # 得出最终概率向量
        doc_vector = np.array(temp_doc_vector) / sum(temp_doc_vector)

        temp_word_vector = []
        # word score 背景噪声向量
        word_bg_vector = np.array([1.0 / num_topn] * num_topn)
        # 得出 word score 向量
        for item in LDA.get_topic_terms(i, topn=num_topn):
            temp_word_vector.append(item[1])
        word_vector = np.array(temp_word_vector)
        
        # 计算分数并放到数组中
        doc_score = asymmetricKL(doc_vector, doc_bg_vector)
        word_score = asymmetricKL(word_vector, word_bg_vector)
        final_score = final_a * doc_score + final_b + word_score
        # print i, final_score, doc_score, word_score
        topic_rank.append((i, final_score))

    # 根据分值进行排序输出
    # 对每个字典进行排序
    ranks = sorted(topic_rank, key=lambda d:-d[1])

    for i in range(len(ranks)):
        line =  LDA.print_topic(ranks[i][0], topn=num_topn)
        line1 = "[Rank %d Topic %d SubRank %d SubTopic %d] %s" % ( rankid, topicid, i, ranks[i][0], clean_topic(line))
        line2 = "分值 %f 共 %d 条 query" % (ranks[i][1], len(topic_result[ranks[i][0]]))
        print line1
        print line2
    
        # 根据概率排序
        result = sorted(topic_result[ranks[i][0]].items(), key=lambda d:-d[1])

        # 写入到对应文本中
        # 这里应该按照顺序写入
        filepath = "%srank%d-%d.txt" % (lda_result_dir, rankid, i)
        with codecs.open(filepath, "w+", "utf-8") as f:
            f.write("%s\n%s\n" % (line1, line2))
            f.write("----------------------\n")
            for re in result:
                f.write("%f %s\n" % (re[1], re[0]))
        print "写入 %s 完成" % filepath
        print "----------------------"


def show_lda_document():
    # 载入词典
    dictionary = corpora.Dictionary.load(dict_path)
    print "载入词典完成"

    # 载入语料
    texts = get_texts()

    corpus = [dictionary.doc2bow(text) for text in texts]
    print "Corpus 数量 %d" % len(corpus)

    LDA = models.LdaModel.load(lda_model)
    print "载入 LDA 模型完成"
    
    # 创建不同的数组，根据 topic 来排序
    topic_result = [{} for i in range(num_topic)]
    for i in range(len(texts)):
        if (i % show_count == 0):
            print "正在处理第 %d 行" % i 
        # 获取每个文本的 topics
        topics = LDA.get_document_topics(corpus[i])
        # 这里选择 Top1
        if (len(topics) < 1):
            continue
        # print len(topics), topics[0]
        topic_result[topics[0][0]]["".join(texts[i])] = topics[0][1]

        # 这里是全量数据
        # for j in range(len(topics)):
        #     # 分配到不同的 topics 中
        #     topic_result[topics[j][0]][texts[i]] = topics[j][1]
    

    # 对每个 topic 进行打分
    topic_rank = []
    for i in range(num_topic):
        if (len(topic_result[i]) < 1):
            continue
        temp_doc_vector = []
        # doc score 背景噪声向量
        doc_bg_vector = np.array([1.0 / len(topic_result[i])] * len(topic_result[i]))
        # 得出 doc score 向量
        for record in topic_result[i].items():
            temp_doc_vector.append(record[1]) # 加入各个概率
        # 得出最终概率向量
        doc_vector = np.array(temp_doc_vector) / sum(temp_doc_vector)

        temp_word_vector = []
        # word score 背景噪声向量
        word_bg_vector = np.array([1.0 / num_topn] * num_topn)
        # 得出 word score 向量
        for item in LDA.get_topic_terms(i, topn=num_topn):
            temp_word_vector.append(item[1])
        word_vector = np.array(temp_word_vector)
        
        # 计算分数并放到数组中
        doc_score = asymmetricKL(doc_vector, doc_bg_vector)
        word_score = asymmetricKL(word_vector, word_bg_vector)
        final_score = final_a * doc_score + final_b + word_score
        # print i, final_score, doc_score, word_score
        topic_rank.append((i, final_score))

    # 根据分值进行排序输出
    # 对每个字典进行排序
    ranks = sorted(topic_rank, key=lambda d:-d[1])

    for i in range(len(ranks)):
        line =  LDA.print_topic(ranks[i][0], topn=num_topn)
        line1 = "[Rank %d Topic %d] %s" % (i, ranks[i][0], clean_topic(line))
        line2 = "分值 %f 共 %d 条 query" % (ranks[i][1], len(topic_result[ranks[i][0]]))
        print line1
        print line2
    
        # 根据概率排序
        result = sorted(topic_result[ranks[i][0]].items(), key=lambda d:-d[1])

        # 写入到对应文本中
        # 这里应该按照得分顺序写入，加一个 - 用来排序
        filepath = "%srank%d-.txt" % (lda_result_dir, i)
        with codecs.open(filepath, "w+", "utf-8") as f:
            f.write("%s\n%s\n" % (line1, line2))
            f.write("----------------------\n")
            for re in result:
                f.write("%f %s\n" % (re[1], re[0]))
        print "写入 %s 完成" % filepath
        print "----------------------"

        # 读入这个分类下的子分类
        # show_lda_topic_document(i, ranks[i][0])

def show_lsi_document():
     # 载入词典
    dictionary = corpora.Dictionary.load(dict_path)
    print "载入词典完成"

    # 载入语料
    texts = get_texts()

    corpus = [dictionary.doc2bow(text) for text in texts]
    print "Corpus 数量 %d" % len(corpus)

    LSI = models.LsiModel.load(lsi_model)
    print "载入 LSI 模型完成"

    filepath = "%slsi-topics.txt" % lsi_result_dir
    with codecs.open(filepath, "w+", "utf-8") as f:
        for i in range(num_topic):
            topic = LSI.show_topic(i, topn=40)
            f.write("---------------------\n")
            f.write("Topic %d\n" % i)
            for term in topic:
                f.write("%s %f\n" % (term[0], term[1]))
    
    # 得到文档的向量，共 num_topic 维，生成测试数据
    print "生成文档向量"
    # 这里的格式需要处理 类型不同
    
    X = np.zeros((len(corpus), num_topic))

    for i in range(len(corpus)):
        vline = LSI[corpus[i]]
        for j in range(len(vline)):
            X[i][j] = vline[j][1]

    print "归一化数据集（特征选择）"
    # normalized dataset for easier parameter selection
    X = StandardScaler().fit_transform(X)

    kmeans = cluster.MiniBatchKMeans(n_clusters=num_topic)
    #affinity_propagation = cluster.AffinityPropagation(damping=.5, preference=None)

    t0 = datetime.datetime.now()
    print "开始 Kmean 聚类，中心个数 %d" % num_topic
    kmeans.fit(X)
    #print "开始 AP 聚类，中心个数 %d" % num_topic
    #affinity_propagation.fit(X)
    t1 = datetime.datetime.now()
    print "聚类耗时", t1-t0
    
    # 输出结果
    print "按照类别写入到结果中"
    y_pred = kmeans.labels_.astype(np.int)
    #y_pred = affinity_propagation.labels_.astype(np.int)

    # 各个类别的结果
    # 也写入到一个统一的文件中
    resultpath = "%sresult_all.txt" % lsi_result_dir
    with codecs.open(resultpath, "w+", "utf-8") as r:
        topic_result = [[] for i in range(num_topic)]
        for i in range (len(texts)):
            topic_result[y_pred[i]].append(texts[i])
        for i in range (len(topic_result)):
            filepath = "%stopic%d.txt" % (lsi_result_dir, i)
            print "写入类别 %d 至 %s" % (i, filepath)
            with codecs.open(filepath, "w+", "utf-8") as f:
                r.write("类别 %d 的记录个数 %d\n" % (i, len(topic_result[i])))
                f.write("类别 %d 的记录个数 %d\n" % (i, len(topic_result[i])))
                for line in topic_result[i]:
                    f.write("".join(line)+"\n")
                    r.write("".join(line)+"\n")
                # 总文件加个分隔符
                r.write("===============================================\n") 
        print "LSI 结果处理完成"

def cluster_query(method):
    # 用来聚类杨老师那边给出的数据
    load_raw_query()
    load_hidden_vector()
    # 检测数量匹配
    if (len(documents) != len(hidden_vectors)):
        print "日志数量与向量数量不符，请检查后重试"
        sys.exit()
    
    # 接下来是正常处理流程
    print "生成隐藏向量数组"
    t0 = datetime.datetime.now()
    X = np.array([[ele for ele in vector[:-1].split("\t")] 
            for vector in hidden_vectors])
    t1 = datetime.datetime.now()
    print "耗时", t1-t0

    # print "归一化数据集（特征选择）"
    # # normalized dataset for easier parameter selection
    # t0 = datetime.datetime.now()
    # X = StandardScaler().fit_transform(X)
    # t1 = datetime.datetime.now()
    # print "耗时", t1-t0

    if (method=="kmeans"):
        print "开始 %s 聚类，中心个数 %d" % (method, num_topic)
        algorithm = cluster.MiniBatchKMeans(n_clusters=num_topic)
    elif (method == "ap"):
        print "开始 %s 聚类，中心个数待定" % method
        algorithm = cluster.AffinityPropagation(damping=.5, preference=None)
    
    t0 = datetime.datetime.now()
    algorithm.fit(X)
    t1 = datetime.datetime.now()
    print "耗时", t1-t0
    
    # 输出结果
    print "按照类别写入到结果中"
    y_pred = algorithm.labels_.astype(np.int)

    # 找到类别中的最大值
    maxY = max(y_pred)
    print "类别个数", maxY+1

    # 各个类别的结果
    topic_result = [[] for i in range(maxY+1)]
    for i in range (len(documents)):
        topic_result[y_pred[i]].append(documents[i])
    for i in range (len(topic_result)):
        filepath = "%stopic%d.txt" % (kmeans_result_dir, i)
        # print "写入类别 %d 至 %s" % (i, filepath)
        with codecs.open(filepath, "w", "utf-8") as f:
            f.write("类别 %d 的记录个数 %d\n" % (i, len(topic_result[i])))
            for line in topic_result[i]:
                f.write(line)
    print "聚类结果处理完成"

def help():
    print "用户问句聚类测试"
    print "用法 python topic_cluster.py dict|tlda|tlsi|slda|slsi|kmeans|ap|help"
    print "dict - 生成词典并保存到 %s 中，原始分词数据在 %s 中" % (dict_path, tokenized_query)
    print "tlda - 用 LDA 模型生成 %d 个 topic，模型保存在 %s 中" % (num_topic, lda_model)
    print "tlsi - 用 LSI 模型生成 %d 个 topic，模型保存在 %s 中" % (num_topic, lsi_model)
    print "slda - 统计 LDA 聚类出来的 topic 并保存在 %s 中" % lda_result_dir
    print "slsi - 统计 LSI 聚类出来的 topic 并保存在 %s 中" % lsi_result_dir
    print "kmeans -  利用 Kmeans 聚类结果并保存在 %s 中" % kmeans_result_dir
    print "ap - 利用 AP 聚类结果并保存在 %s 中" % ap_result_dir

if __name__=="__main__":
    if len(argv) != 2:
        help()
        sys.exit()
    if (argv[1] == "dict"):
        tokenize_query()
        generate_dict() 
    elif (argv[1] == "tlda"):
        load_tokenized_query()
        generate_lda_topic() 
    elif (argv[1] == "tlsi"):
        load_tokenized_query()
        generate_lsi_topic() 
    elif (argv[1] == "slda"):
        load_tokenized_query()
        show_lda_document()
    elif (argv[1] == "slsi"):
        load_tokenized_query()
        show_lsi_document()
    elif (argv[1] == "kmeans"):
        cluster_query(argv[1])
    elif (argv[1] == "ap"):
        cluster_query(argv[1])
    elif (argv[1] == "help"):
        help()
    else:
        print "未知命令"
        help()
