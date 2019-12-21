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
        content     = document.read().lower()
        words       = re.findall(regex, content)
        Document[fileName] = words
        Vocabulary.update(words)

    Vocabulary2 = set([])
    for w in Vocabulary:
        if len(w)>=10:
            Vocabulary2.add(w)
    Vocabulary = Vocabulary2
    print(Vocabulary)

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
    for w1 in range(0, V-1): 
        for w2 in range(w1+1, V): 
            dist[w1,w2] = numpy.linalg.norm(M[w1,:]-M[w2,:])
    print(dist)


    # criando o grafo de documentos (similaridade entre documentos)
    dist = 1 - (dist-numpy.nanmin(dist))/(numpy.nanmax(dist)-numpy.nanmin(dist)) 

    txtGraph = "\ngraph{"
    for w1 in range(0, V-1): 
        for w2 in range(w1+1, V): 
          if dist[w1,w2]!=numpy.nan and dist[w1,w2]>=0.95:
              txtGraph += '\n "{0}" -- "{1}"[label="{2:.2f}", penwidth={2:.2f}]'. format(vocabulary[w1], vocabulary[w2], dist[w1,w2] )
    txtGraph += "\n}"
   
    print(txtGraph)


   


