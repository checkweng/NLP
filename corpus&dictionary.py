# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:25:59 2019

@author: Administrator
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
