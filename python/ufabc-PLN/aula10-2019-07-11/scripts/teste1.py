# coding=utf8
import sys
import re
import os
import numpy
import matplotlib.pyplot as plt

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+" 

if __name__ == '__main__':
    dirDB = sys.argv[1]
    
    Document   = dict([])
    Vocabulary = set([])

    # leitura das stopwords
    Stopwords       = set([]) 
    for s in open("stopwords-pt.txt",'r').readlines():
        Stopwords.add(s.strip().lower())

    # leitura dos documentos
    for fileName in os.listdir(dirDB):
        Document[fileName] = []
        document = open(dirDB+"/"+fileName,'r')
        content  = document.read().lower()

        for w in re.findall(regex, content):
            if w not in Stopwords and len(w)>=3:
                Document[fileName].append(w)
        Vocabulary.update( Document[fileName] )


    D = len(Document)
    V = len(Vocabulary)
    S = len(Stopwords)

    # contabilizando os pares de palavras
    k           = 3
    Mcontext    = numpy.zeros((V, V))
    iVocabulary = dict([])
    
    for (i,w) in enumerate(Vocabulary):
        iVocabulary[w] = i

    for d in Document.keys():
        print (d)
        for (i,w) in enumerate(Document[d]):
            context = []
            if i>k:
                context += Document[d][i-k:i]
            if i<len(Document[d])-k:
                context += Document[d][i+1:i+k+1]

            print (i, w, context)

            iw = iVocabulary[w]
            for wc in context:
                Mcontext[iw, iVocabulary[wc]] += 1

    # Informacoes basicas
    print("Numero de documentos  : {}".format( D ))
    print("Tamanho do vocabulario: {}".format( V ))
    print("Numero de stopwords:    {}".format( S ))
    plt.imshow(Mcontext[1:200,1:200], cmap='binary')
    plt.show()

