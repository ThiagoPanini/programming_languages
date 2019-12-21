# coding=utf8
import sys
import re

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+"   # raw string
#regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ0-9]+"   # raw string


if __name__ == '__main__':
    fileName = sys.argv[1]

    # leitura do documento
    document = open(fileName,'r')
    content  = document.read()  # devolve um vetor contendo as linhas do arquivo
    content  = content.lower()

    Words    = re.findall(regex, content)

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

    # Testes 
    phrases = ["ele é",
               "ele é uma", 
               "ele é uma pessoa", 
               "ele é uma pessoa de", 
               "ele é uma pessoa de verdade"]
   
    for phrase in phrases:
        Words = re.findall(regex, phrase)
        P = float(1.0)
        for i in range(0, len(Words)-1):
            P = P * Bigrams[ (Words[i],Words[i+1]) ] / Unigrams[Words[i]]

        print ( "{1:.20f} : Frase: {0}".format( phrase, P ) )   
