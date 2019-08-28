# 引入 word2vec
from gensim.models import word2vec

# 引入日志配置
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 引入数据集
raw_sentences = ["the quick brown fox jumps over the lazy dogs","yoyoyo you go home now to sleep"]

# 切分词汇
sentences= [s.encode('utf-8').split() for s in sentences]

# 构建模型
model = word2vec.Word2Vec(sentences, min_count=1)

# 进行相关性比较
model.similarity('dogs','you')


##或者采用先建立词典再train的过程
model = gensim.models.Word2Vec(iter=1)  # an empty model, no training yet
model.build_vocab(some_sentences)  # can be a non-repeatable, 1-pass generator
model.train(other_sentences)  # can be a non-repeatable, 1-pass generator


##外部语料集  这里语料集中的语句是经过分词的

sentences = word2vec.Text8Corpus('text8')
model = word2vec.Word2Vec(sentences, size=200)

##模型保存与读取

model.save('text8.model')

model1 = Word2Vec.load('text8.model')

model.save_word2vec_format('text.model.bin', binary=True)

model1 = word2vec.Word2Vec.load_word2vec_format('text.model.bin', binary=True)

##模型预测
model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)

model.doesnt_match("breakfast cereal dinner lunch";.split())

model.similarity('woman', 'man')

model.most_similar(['man'])

##如果我们希望直接获取某个单词的向量表示，直接以下标方式访问即可

model['computer']

##模型评估

model.accuracy('/tmp/questions-words.txt')
