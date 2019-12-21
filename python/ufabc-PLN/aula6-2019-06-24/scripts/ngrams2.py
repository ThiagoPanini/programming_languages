# coding=utf8
import sys
import re

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+"   # raw string

if __name__ == '__main__':
    fileName = sys.argv[1]

    # leitura do documento
    document = open(fileName,'r')
    content  = document.read()
    content  = content.lower()

    Words    = re.findall(regex, content)
    
    #stopwordsfile = open("stopwords.txt",'r')
    #stopwords       = set([])            
    #for s in stopwordsfile.readlines():                                                                                                                                                     
    #    stopwords.add(s.strip().lower())
    #
    #Words2 = list([])
    #for w in Words:
    #    if w not in stopwords:
    #        Words2.append(w)
    #Words = Words2


    # Unigramas
    Unigrams = dict([])
    for i, w in enumerate(Words):
        if w not in Unigrams:
            Unigrams[w] = 0
        Unigrams[w] += 1

    # Bigramas
    Bigrams = dict([])
    for i in range(0, len(Words)-1):
        b = (Words[i], Words[i+1])
        if b not in Bigrams:
            Bigrams[b] = 0
        Bigrams[b] += 1

    BigramProbabilities = dict([])
    for (w1,w2) in Bigrams.keys():
        BigramProbabilities[ (w1,w2) ] = Bigrams [ (w1,w2) ] / Unigrams[ w1 ]


    # Testes 
    while True:
        phrase = input("\nDigite uma frase: ")
        Words = re.findall(regex, phrase)

        wNext     = ""
        wNextProb = 0

        for (w1,w2) in BigramProbabilities.keys():
            if w1 == Words[-1] and BigramProbabilities[(w1,w2)] > wNextProb:
                wNext     = w2
                wNextProb = BigramProbabilities[(w1,w2)]

        print ( "{0} -> {1} ({2:.2f}%)".format( phrase, wNext.upper(), wNextProb*100 ) )   
