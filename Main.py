'''
POS tag list
CC coordinating conjunction
CD cardinal digit
DT determiner
EX existential there (like: “there is” … think of it like “there exists”)
FW foreign word
IN preposition/subordinating conjunction
JJ adjective ‘big’
JJR adjective, comparative ‘bigger’
JJS adjective, superlative ‘biggest’
LS list marker 1)
MD modal could, will
NN noun, singular ‘desk’
NNS noun plural ‘desks’
NNP proper noun, singular ‘Harrison’
NNPS proper noun, plural ‘Americans’
PDT predeterminer ‘all the kids’
POS possessive ending parent’s
PRP personal pronoun I, he, she
PRP$ possessive pronoun my, his, hers
RB adverb very, silently,
RBR adverb, comparative better
RBS adverb, superlative best
RP particle give up
TO, to go ‘to’ the store.
UH interjection, errrrrrrrm
VB verb, base form take
VBD verb, past tense took
VBG verb, gerund/present participle taking
VBN verb, past participle taken
VBP verb, sing. present, non-3d take
VBZ verb, 3rd person sing. present takes
WDT wh-determiner which
WP wh-pronoun who, what
WP$ possessive wh-pronoun whose
WRB wh-abverb where, when
'''
import pandas as pd
import numpy as np
from numpy import *
import json
import gensim
from gensim import corpora
from gensim.models import LsiModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.collocations import *
from nltk.corpus import wordnet as wn
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import brown
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.chunk import api
from nltk.chunk.api import ChunkParserI
from nltk import RegexpParser
import scipy
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
from nltk.tokenize import PunktSentenceTokenizer
from smart_open import smart_open
import os
import os.path
from pprint import pprint

with open('test1-300.json', 'r') as f:
    jsonData = json.load(f)

dict = {}
for i in jsonData["Values"]:
    #if (i["overall"] > 1) and (i["overall"] < 5):
    if (i["overall"] > 2) and (i["overall"] < 5):
         tokenizer = PunktSentenceTokenizer()
         tok_text = tokenizer.tokenize(i["reviewText"])
         for k in tok_text:
              words = word_tokenize(k)
              pos_text = nltk.pos_tag(words)
              ChunkGramma = r"""Chunk: {<VB.?>*<RB>*<JJ.?>+<NN.?>?} 
                                       {<NN.?>?<VB.?>+<RB>+}
                                       {<NN.?>?<VB.?>+<RB>?<JJ.?>+}     
                                       {<NN.?>?<VB.?>+<JJ.?>+}   
                             """
              chunkParser = nltk.RegexpParser(ChunkGramma)
              chunked = chunkParser.parse(pos_text)
              #chunked.draw()
         customStopWords = set(stopwords.words('english') + list(punctuation))
         WordsStopResult = [word for word in pos_text if word not in customStopWords]
         #lemmitazer = WordNetLemmatizer()
         #lemmitazer_output = ' '.join(lemmitazer.lemmatize(word) for word in WordsStopResult)
         dict.update({i["id"]:WordsStopResult})

for (key,value) in dict.items() :
    print(key,value)

number_of_topics = 3
words = 20
stemmer = SnowballStemmer('english')

dict1 = {}
for i in jsonData:
        if i["overall"] == 5.0:
             tok_text = word_tokenize(i["reviewText"])
             customStopWords = set(stopwords.words('english') + list(punctuation))
             WordsStopResult = [word for word in tok_text if word not in customStopWords]
             lemmitazer = WordNetLemmatizer()
             lemmitazer_output = ' '.join(lemmitazer.lemmatize(word) for word in WordsStopResult)
             dict1.update({i["id"]: WordsStopResult})

with open('Dict.txt', 'w') as Dict:
     for key,val in dict1.items():
         Dict.write('{}:{}\n'.format(key,val)) # Dict.txt - file with clear review's text from future recycle

mydict = corpora.Dictionary(simple_preprocess(line,deacc=True) for line in open('Dict.txt', encoding='utf-8'))
#print(mydict)
corpus = [mydict.doc2bow(simple_preprocess(line)) for line in open('Dict.txt', encoding='utf-8')]
#print (corpus)

lsamodel = LsiModel(corpus, num_topics=number_of_topics, id2word=mydict)  # train model
# print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
tx=lsamodel.print_topics()
pprint(tx, open("Topic.txt", "w"))#Topic.txt - file with main thems(parametrs) from reviews, which have rating 5.0

# ----------------------------------create vectors with word's index and tf-idf recycling------------------------------
with open("Dict.txt", "r") as file:
    documents = file.read().splitlines()
# print(documents)
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()
bag_of_words = count_vectorizer.fit_transform(documents)
feature_names = count_vectorizer.get_feature_names()
pprint(pd.DataFrame(bag_of_words.toarray(), columns=feature_names), open("matrix.txt", "w"))
# with open("matrix.txt","w") as f: #вывод проиндексированной матрицы в файл
#     for i in range(len(bag_of_words.toarray())):
#         for j in range(len(bag_of_words.toarray()[i])):
#             f.write(str(bag_of_words.toarray()[i][j]))
#         f.write("\n")

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
values = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names()
print(pd.DataFrame(values.toarray(), columns=feature_names))
# with open("matrix.txt","w") as f:
#     for i in range(len(values.toarray())):
#         for j in range(len(values.toarray()[i])):
#             f.write(str(values.toarray()[i][j]))
#         f.write("\n")

dict2 = {}
for i in jsonData["Values"]:
        if i["overall"] == 1.0:
             tok_text = word_tokenize(i["reviewText"])
             customStopWords = set(stopwords.words('english') + list(punctuation))
             WordsStopResult = [word for word in tok_text if word not in customStopWords]
             lemmitazer = WordNetLemmatizer()
             lemmitazer_output = ' '.join(lemmitazer.lemmatize(word) for word in WordsStopResult)
             dict2.update({i["id"]: WordsStopResult})

with open('Dict_2.txt', 'w') as Dict_2:
     for key,val in dict2.items():
         Dict_2.write('{}:{}\n'.format(key,val)) # Dict.txt - file with clear review's text from future recycle

mydict_2 = corpora.Dictionary(simple_preprocess(line,deacc=True) for line in open('Dict_2.txt', encoding='utf-8'))
#print(mydict)
corpus_2 = [mydict_2.doc2bow(simple_preprocess(line)) for line in open('Dict_2.txt', encoding='utf-8')]
#print (corpus)

lsamodel_2 = LsiModel(corpus_2, num_topics=number_of_topics, id2word=mydict_2)  # train model
# print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
tx=lsamodel_2.print_topics()
pprint(tx, open("Topic_2.txt", "w"))#Topic.txt - file with main thems(parametrs) from reviews, which have rating 1.0

# d={};c=[]
# for i in range(1,len(dict)):
#     word = dict.get(i)
#     if word is not None:
#         word_stem=[stemmer.stem(w).lower() for w in word if len(w) > 1 and  w.isalpha()]
#         word_stop=[ w for w in word_stem if w not in WordsStopResult]
#         for w in word_stop:
#             if w not in c:
#                 c.append(w)
#                 d[w]= [i]
#             elif w in c:
#                 d[w]= d[w]+[i]
#     a=len(c); b=len(dict)
#     A = np.zeros([a,b])
#     c.sort()
#     for i, k in enumerate(c):
#         for j in d[k]:
#             A[i,j] += 1
#
# def printMatrix ( matrix ):
#    for i in range ( len(matrix) ):
#        print(c[i] , end = " ")
#        for j in range ( len(matrix[i]) ):
#           print (matrix[i][j], end = " " )
#        print ()
#
# printMatrix(A)
#
# wpd = sum(A, axis=1)
# dpw= sum(asarray(A > 0,'i'), axis=1)
# rows, cols = A.shape
# for i in range(rows):
#     for j in range(cols):
#              m=float(A[i,j])/wpd[j]
#              n=log(float(cols) /dpw[i])
#              A[i,j] =round(n*m,2)
#
# U, S,Vt = np.linalg.svd(A)
# rows, cols = U.shape
# for j in range(0,cols):
#            for i  in range(0,rows):
#                U[i,j]=round(U[i,j],4)
# print('Первые 2 столбца ортогональной матрицы U слов')
# for i, row in enumerate(U):
#     print(c[i], row[0:2])
# res1=-1*U[:,0:1]; res2=-1*U[:,1:2]
# data_word=[]
# for i in range(0,len(c)):# Подготовка исходных данных в виде вложенных списков координат
#     data_word.append([res1[i][0],res2[i][0]])
# plt.figure()
# plt.subplot(221)
# dist = pdist(data_word, 'euclidean')# Вычисляется евклидово расстояние (по умолчанию)
# plt.hist(dist, 500, color='green', alpha=0.5)# Диаграмма евклидовых расстояний
# Z = hierarchy.linkage(dist, method='average')# Выделение кластеров
# plt.subplot(222)
# hierarchy.dendrogram(Z, labels=c, color_threshold=.25, leaf_font_size=8, count_sort=True,orientation='right')
# print('Первые 2 строки ортогональной матрицы Vt документов')
# rows, cols = Vt.shape
# for j in range(0,cols):
#     for i  in range(0,rows):
#         Vt[i,j]=round(Vt[i,j],4)
# print(-1*Vt[0:2, :])
# res3=(-1*Vt[0:1, :]);res4=(-1*Vt[1:2, :])
# data_docs=[];name_docs=[]
# for i in range(0,len(dict)):
#     name_docs.append(str(i))
#     data_docs.append([res3[0][i],res4[0][i]])
# plt.subplot(223)
# dist = pdist(data_docs, 'euclidean')
# plt.hist(dist, 500, color='green', alpha=0.5)
# Z = hierarchy.linkage(dist, method='average')
# plt.subplot(224)
# hierarchy.dendrogram(Z, labels=name_docs, color_threshold=.25, leaf_font_size=8, count_sort=True)
#
# #plt.show()
# print('Первые 3 столбца ортогональной матрицы U слов')
# for i, row in enumerate(U):
#     print(c[i], row[0:3])
# res1=-1*U[:,0:1]; res2=-1*U[:,1:2];res3=-1*U[:,2:3]
# data_word_xyz=[]
# for i in range(0,len(c)):
#     data_word_xyz.append([res1[i][0],res2[i][0],res3[i][0]])
# plt.figure()
# plt.subplot(221)
# dist = pdist(data_word_xyz, 'euclidean')# Вычисляется евклидово расстояние (по умолчанию)
# plt.hist(dist, 500, color='green', alpha=0.5)#Диаграмма евклидовых растояний
# Z = hierarchy.linkage(dist, method='average')# Выделение кластеров
# plt.subplot(222)
# hierarchy.dendrogram(Z, labels=c, color_threshold=.25, leaf_font_size=8, count_sort=True,orientation='right')
# print('Первые 3 строки ортогональной матрицы Vt документов')
# rows, cols = Vt.shape
# for j in range(0,cols):
#     for i  in range(0,rows):
#         Vt[i,j]=round(Vt[i,j],4)
# print(-1*Vt[0:3, :])
# res3=(-1*Vt[0:1, :]);res4=(-1*Vt[1:2, :]);res5=(-1*Vt[2:3, :])
# data_docs_xyz=[];name_docs_xyz=[]
# for i in range(0,len(dict)):
#     name_docs_xyz.append(str(i))
#     data_docs_xyz.append([res3[0][i],res4[0][i],res5[0][i]])
# plt.subplot(223)
# dist = pdist(data_docs_xyz, 'euclidean')
# plt.hist(dist, 500, color='green', alpha=0.5)
# Z = hierarchy.linkage(dist, method='average')
# plt.subplot(224)
# hierarchy.dendrogram(Z, labels=name_docs_xyz, color_threshold=.25, leaf_font_size=8, count_sort=True)
# plt.show()


# stop_words = stopwords.words('english')
# class BoWCorpus(object):
#     def __init__(self, path, dictionary):
#         self.filepath = path
#         self.dictionary = dictionary
#
#     def __iter__(self):
#         global mydict  # OPTIONAL, only if updating the source dictionary.
#         for line in smart_open(self.filepath, encoding='latin'):
#             # tokenize
#             tokenized_list = simple_preprocess(line, deacc=True)
#             # create bag of words
#             bow = self.dictionary.doc2bow(tokenized_list, allow_update=True)
#             # update the source dictionary (OPTIONAL)
#             mydict.merge_with(self.dictionary)
#             # lazy return the BoW
#
# mydict = corpora.Dictionary(simple_preprocess(line,deacc=True) for line in open('Dict.txt', encoding='utf-8'))
# corpus = [mydict.doc2bow(simple_preprocess(line)) for line in open('Dict.txt', encoding='utf-8')]

# for line in corpus:
# print(line)
# ------------------------подсчет_недовольных
# for doc in corpus:
#     print([[mydict[id], freq] for id, freq in doc])
#
# from gensim import models
# tfidf = models.TfidfModel(corpus, smartirs='ntc')
#
# for doc in tfidf[corpus]:
#     print([[mydict[id], np.around(freq, decimals=2)] for id, freq in doc])