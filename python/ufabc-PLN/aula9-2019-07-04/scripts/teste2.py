# coding=utf8
import sys
import re
import os
import numpy

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+" 

if __name__ == '__main__':
    dirDB = sys.argv[1]
    
    Document   = dict([])
    Vocabulary = set([])


    # leitura dos documentos
    for fileName in os.listdir(dirDB):
        document    = open(dirDB+"/"+fileName,'r')
        content     = document.read()
        words       = re.findall(regex, content)
        Document[fileName] = words
        Vocabulary.update(words)
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


    # distancia entre documentos
    dist = numpy.ones((D,D))*numpy.nan
    for d1 in range(0, D-1): 
        for d2 in range(d1+1, D): 
            # distancia Euclidiana
            #dist[d1,d2] = numpy.linalg.norm(M[:,d1]-M[:,d2])
            # distancia (similaridade) Cosseno
            dist[d1,d2] = numpy.dot(M[:,d1], M[:,d2])/(numpy.linalg.norm(M[:,d1])*numpy.linalg.norm(M[:,d2]))

    print(dist)


    # normalizando a distância: 1: maior distância, 0: menor distância
    dist = 1 - (dist-numpy.nanmin(dist))/(numpy.nanmax(dist)-numpy.nanmin(dist)) 


    # criando o grafo de documentos (similaridade entre documentos)
    txtGraph = "\ngraph{"
    for d1 in range(0, D-1): 
        for d2 in range(d1+1, D): 
          #if dist[d1,d2]!=numpy.nan :
          if dist[d1,d2]!=numpy.nan and dist[d1,d2]>=0.5:
              txtGraph += '\n "{0}" -- "{1}"[label="{2:.2f}", penwidth={2:.2f}]'. format(documents[d1], documents[d2], dist[d1,d2] )
    txtGraph += "\n}"
   
    print(txtGraph)


   


