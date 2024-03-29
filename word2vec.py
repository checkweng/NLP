# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:40:27 2019

@author: ny002
"""

import jieba.analyse
import codecs

#以写的方式打开原始的简体中文语料库
f=codecs.open('zhwiki_jian_zh.txt','r',encoding="utf8")
#将分完词的语料写入到wiki_jian_zh_seg-189.5.txt文件中
target = codecs.open("wiki_jian_zh_seg-189.5.txt", 'w',encoding="utf8")

print('open files')
line_num=1
line = f.readline()

#循环遍历每一行，并对这一行进行分词操作
#如果下一行没有内容的话，就会readline会返回-1，则while -1就会跳出循环
while line:
    print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()

#关闭两个文件流，并退出程序
f.close()
target.close()
exit()

##########word2vec过程


import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

#程序的入口
#1.如果当前脚本文件做模块供其他程序使用的话，不会执行if __name__ == '__main__':中的内容
#2.如果直接执行当前的额脚本文件的话，执行if __name__ == '__main__':中的内容
if __name__ == '__main__':

    #1.os.path.basename('g://tf/code') ==>code
    #2.sys.argv[0]获取的是脚本文件的文件名称
    program = os.path.basename(sys.argv[0])
    #指定name，返回一个名称为name的Logger实例
    logger = logging.getLogger(program)
    #1.format: 指定输出的格式和内容，format可以输出很多有用信息，
    #%(asctime)s: 打印日志的时间
    #%(levelname)s: 打印日志级别名称
    #%(message)s: 打印日志信息
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    #打印这是一个通知日志
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    #inp:分好词的文本
    #outp1:训练好的模型
    #outp2:得到的词向量
    inp, outp1, outp2 = sys.argv[1:4]
    '''
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度； 
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词； 
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃； 
    workers：是训练的进程数（需要更精准的解释，请指正），默认是当前运行机器的处理器核数。这些参数先记住就可以了。
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    alpha (float, optional) – 初始学习率
    iter (int, optional) – 迭代次数，默认为5
    '''
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    #不以C语言可以解析的形式存储词向量
    model.wv.save_word2vec_format(outp2, binary=False)


##测试
    
from gensim.models import Word2Vec

en_wiki_word2vec_model = Word2Vec.load('wiki_zh_jian_text.model')

testwords = ['金融','上','股票','跌','经济']
for i in range(5):
    res = en_wiki_word2vec_model.most_similar(testwords[i])
    print (testwords[i])
    print (res)

