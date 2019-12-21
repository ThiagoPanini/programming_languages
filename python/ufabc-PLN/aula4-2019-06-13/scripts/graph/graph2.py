# coding=utf8
import sys
import re

#regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+"   # raw string
regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"   # raw string


if __name__ == '__main__':
    fileName = sys.argv[1]
    weight   = int(sys.argv[2])

    # leitura das stopwords
    stopwordsfile = open("stopwords.txt",'r')
    stopwords       = set([]) 
    for s in stopwordsfile.readlines():
        stopwords.add(s.strip().lower())

    # leitura do documento
    document = open(fileName,'r')
    content  = document.read()  # devolve o conteudo do arquivo
    
    Words    = re.findall(regex, content)
    Edges    = dict([])

    # contando a frequencia dos pares de palavras
    for i in range(0, len(Words)-1):
        if Words[i] not in stopwords and Words[i+1] not in stopwords:
            edge = (Words[i], Words[i+1])
            if edge not in Edges:
                Edges[edge] = 0
            Edges[edge] += 1

    # criando o grafo direcionado (digraph)
    txtGraph = "\ndigraph{"
    for v in Edges.keys():
        if Edges[v]>=weight:
            txtGraph += '\n "{}" -> "{}"[label="{}"]'. format(v[0], v[1], Edges[v])
    txtGraph += "\n}"

    print(txtGraph)
    print ( "\nQuantidade de palavras: {}".format(len(Words)) )   
    print ( "Quantidade de arestas : {}".format(len(Edges)) )   

