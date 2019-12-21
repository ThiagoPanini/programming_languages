import unicodedata
import os, sys
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(2018)
import nltk

def lemmatize_stemming(text):
    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

data = pd.read_csv('teste.csv', sep='\t', lineterminator='\n')
data_text = data[['AB']]
data_text['index'] = data_text.index
documents = data_text
print(len(documents))
print(documents[:20])

doc_sample = documents[documents['index'] == 0].values[0][0]
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

#****Grava as palavras de cada resumo ja pre-processadas****
processed_docs = documents['AB'].map(preprocess)

ARQ1 = open('words.txt','w')
for words in processed_docs[:len(processed_docs)]:
    c=0
    for word in words:
        if(c<len(words) and c!=0):ARQ1.write(", ")
        ARQ1.write("%s"%word)
        c+=1
    ARQ1.write('\n')
ARQ1.close()

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=1, no_above=1.0, keep_n=20)
count = 0
ARQ = open('dictionary.txt','w')
for k, v in dictionary.iteritems():
    ARQ.write("%s (%d)\n"%(v,k))
    print(k, v)
ARQ.close()

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[0])

from gensim import corpora, models
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=2, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

ARQ = open('classification.txt','w')
for j in range(len(bow_corpus)):
    c=0
    for index, score in sorted(lda_model[bow_corpus[j]], key=lambda tup: -1*tup[1]):
        if(c==0):ARQ.write("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))
        c+=1
    ARQ.write('\n')
ARQ.close()
