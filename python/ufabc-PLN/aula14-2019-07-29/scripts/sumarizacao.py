#!/usr/bin/env python
import sys
import nltk

if __name__ == '__main__':
    fileName = sys.argv[1]    
    document = open(fileName,'r')
    content  = document.read()
    ranking  = []

    for (i, sentence) in enumerate(nltk.sent_tokenize(content)):
        tokens         = nltk.word_tokenize(sentence) 
        tagged         = nltk.pos_tag(tokens) 
        named_entities = nltk.ne_chunk(tagged, binary=True) 
        nTokens        = len(tokens)

        print (f"\n{i} : >>{sentence}<<\n")

        # Contagem dos substantivos: NN e NNP.
        nNouns = 0
        for (word, pos) in tagged:
            if pos in ["NN", "NNP"]:
                nNouns += 1
                print (word, pos)

        # Contagem de entidades nomeadas
        nNes = 0
        for ne in named_entities:
            if hasattr(ne, 'label') and ne.label() == "NE":
                nNes += 1
                print (ne)

        # Cálculo do score ingenuo
        score = (nNes + nNouns)/float(nTokens)
        ranking.append( (i, score, nNes, nNouns, nTokens, sentence) )

    # Imprimindo as frases em ordem invertida pelo score.
    for r in sorted( ranking, key=lambda x: x[1], reverse=True):
        print (r)


