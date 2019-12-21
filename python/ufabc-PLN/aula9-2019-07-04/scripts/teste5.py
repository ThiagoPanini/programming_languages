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


    # leitura dos documentos
    for fileName in os.listdir(dirDB):
        document    = open(dirDB+"/"+fileName,'r')
        content     = document.read().lower()
        words       = re.findall(regex, content)
        Document[fileName] = words
        Vocabulary.update(words)

    Vocabulary2 = set([])
    for w in Vocabulary:
        if len(w)>=10:
            Vocabulary2.add(w)
    Vocabulary = Vocabulary2

    D = len(Document)
    V = len(Vocabulary)
    print("Numero de documentos  : {}".format( D ))
    print("Tamanho do vocabulario: {}".format( V ))


    # calculando as frequencias das palavras nas obras
    M          = numpy.zeros((V, D))
    documents  = list(Document.keys())
    vocabulary = list(Vocabulary)
    for j in range(0, D):
        d = documents[j]       
        print (d)
        for i in range(0, V):
            w      = vocabulary[i]       
            M[i,j] = Document[d].count(w)


    # distancia entre palavras
    dist = numpy.ones((V,V))*numpy.nan
    for w2 in range(0, V-1): 
        for w1 in range(w2+1, V): 
            dist[w1,w2] = numpy.linalg.norm(M[w1,:]-M[w2,:])
    print(dist)


    # similaridade entre documentos
    dist = 1 - (dist-numpy.nanmin(dist))/(numpy.nanmax(dist)-numpy.nanmin(dist)) 

    plt.imshow(dist, cmap='Greens', interpolation='nearest')
    plt.yticks(range(0,len(vocabulary)), vocabulary )
    plt.xticks(range(0,len(vocabulary)), vocabulary )
    plt.xticks(rotation=90)
    plt.show()



   


