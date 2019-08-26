# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:25:59 2019

@author: Administrator
corpus 就是词袋
"""



import logging
from gensim import corpora
import re
import jieba
from collections import defaultdict
from pprint import pprint  # pretty-printer
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)




documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]

stoplist = set('for a of the and to in'.split()) 
texts = [[word for word in document.lower().split() if word not in stoplist]
         for document in documents]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
        
texts = [[token for token in text if frequency[token] > 1]   ##筛选出词频率大于1的词
         for text in texts]


dictionary = corpora.Dictionary(texts) # 将文档存入字典

get_ipython().magic('pinfo2 dictionary')

dictionary.token2id   # 单次词频键值对，可以直接用来生成词云 词典的词有哪些

dictionary.dfs #词典的健值与词频的对应关系

dictionary.filter_tokens()

dictionary.compactify()

dictionary.save('d:/deerwester.dict')

dictionary.doc2bow()##对新词进行翻译


# 输出dictionary中个单词的出现频率
def PrintDictionary():
    token2id = dictionary.token2id
    dfs = dictionary.dfs
    token_info = {}
    for word in token2id:
        token_info[word] = dict(
            word = word,
            id = token2id[word],
            freq = dfs[token2id[word]]
        )
    token_items = token_info.values()
    token_items = sorted(token_items, key = lambda x:x['id'])  ##按其中的x的id进行排序
    print('The info of dictionary: ')
    pprint(token_items)


# 测试 ditonary的doc2bow功能，转化为one-hot presentation
new_doc = "Human computer interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec)  # interaction" 没有在字典中，所以忽略了

corpus = [dictionary.doc2bow(text) for text in texts]  ##形成词袋

corpus  # 词带中对应词出现的次数，如(2, 1)代表词带中的2号词出现了1次

corpora.MmCorpus.serialize('../../tmp/deerwester.mm', corpus)  # 保存至本地
# 除了MmCorpus以外，还有SvmLightCorpus等以各种格式存入磁盘


###################################################################################################################计算文本相似度
from gensim import corpora, models, similarities
from pprint import pprint


# 读取词典和文档，计算tf idf 和 lsi 的表达方式，并生成相似度矩阵
dictionary = corpora.Dictionary.load('../../tmp/deerwester.dict')
corpus = corpora.MmCorpus('../../tmp/deerwester.mm') # 上一示例中生成的数据
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]  # TIDIF 逆文档频率
lsi_model = models.LsiModel(corpus_tfidf,id2word=dictionary,num_topics=2)
corpus_lsi = lsi_model[corpus_tfidf] # LSI 潜在语义索引
corpus_simi_matrix = similarities.MatrixSimilarity(corpus_lsi)
corpus_simi_matrix.save('../../tmp/deerwester.index')
similarities.MatrixSimilarity.load('../../tmp/deerwester.index') # 下次调用

# 基于tfidf的文本相似度分析 
index = similarities.MatrixSimilarity(corpus_tfidf) 
vec_bow =[dictionary.doc2bow(text) for text in raw_data]   #把用户raw_data语料转为词包
all_reult_sims = []
times_v2 = 0 

###对每个用户预料与标准预料计算相似度

for i in vec_bow:    
     #直接使用上面得出的tf-idf 模型即可得出商品描述的tf-idf 值
    sims = index[tfidf_model[i]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    result_sims = []    
    for i,j in sims:
        result_sims.append([map_value_user[times_v2],map_value[i],j])
    times_v2 += 1
    all_reult_sims.append(result_sims[:20])
print(all_reult_sims)  # 查看前20条显示相似文本

# 基于lsi的文本相似度分析 
index = similarities.MatrixSimilarity(corpus_lsi) 
vec_bow =[dictionary.doc2bow(text) for text in raw_data]  #把用户语料raw_data转为词包
all_reult_sims = []
times_v2 = 0 
for i in vec_bow:    
     #直接使用上面得出的tf-idf 模型即可得出商品描述的tf-idf 值
    sims = index[lsi[tfidf_model[i]]]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    result_sims = []    
    for i,j in sims:
        result_sims.append([map_value_user[times_v2],map_value[i],j])
    times_v2 += 1
    all_reult_sims.append(result_sims[:20])
print(all_reult_sims)  # 查看前20条显示相似文本
