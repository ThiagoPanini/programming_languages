import sys
import re
from collections import defaultdict as ddict

# Programa baseado nas notas do livro texto:
#     Jurafsky, D. & Martin, J. H. (2009). 
#     Speech and language processing: An introduction to natural language processing, computational linguistics, and speech recognition. Pearson/Prentice Hall.

class NGrams(object):

    def __init__(self, max_n, Words=None):
        self.max_n   = max_n
        self.n_range = range(1, max_n+1)  # Para trigramas sera [1, 2, 3]
        self.Counts  = ddict(lambda: 0)   # Colecao com valor padrao igual a zero (estrutura mais importante deste programa)
        
        # Se o conjunto de palavras for indicada
        if Words is not None:
            self.update(Words)


    # Contabilizamos os n-gramas a partir do conjunto de palavras.
    def update(self, Words):
        # Caso especial: tupla vazia (util para o metodo 'probability')
        # O valor eh igual ao numero de palavras
        self.Counts[()] += len(Words)

        # Conta os n-gramas para todos os tamanhos. Para trigramas: (), (w), (w,w), (w,w,w) 
        for i, word in enumerate(Words):
            for n in self.n_range:
                if i+n <= len(Words):
                    self.Counts[tuple( Words[i:i+n] )] += 1

    # Calcula a probabilidade para a frase: Words
    def probability(self, Words):
        if len(Words) <= self.max_n:
            return self._probability(Words)
        else:
            P = 1
            for i in range(len(Words) - self.max_n + 1):
                ngram = Words[i:i + self.max_n]
                P     = P * self._probability(ngram)
            return P

    # Calcula a aproximacao para o n-grama usando seu prefixo
    def _probability(self, ngram):
        ngram        = tuple(ngram)
        ngram_count  = self.Counts[ngram]
        prefix_count = self.Counts[ngram[:-1]]

        # Se uma tupla (n-grama) nao for observada devolvemos zero
        if ngram_count and prefix_count:
            return ngram_count / prefix_count
        else:
            return 0.0


if __name__ == '__main__':
    regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+" 

    fileName = sys.argv[1]
    N        = int(sys.argv[2])  # tamanho do n-grama

    document = open(fileName,'r')
    content  = document.read() 
    content  = content.lower()
    Words    = re.findall(regex, content)

    NG = NGrams(N, Words)
    
    # Testes 
    phrases = ["ele é",
               "ele é uma", 
               "ele é uma pessoa", 
               "ele é uma pessoa de", 
               "ele é uma pessoa de verdade"]

    for phrase in phrases:
        Words = re.findall(regex, phrase)
        print ( "{1:.20f} : Frase: {0}".format( phrase, NG.probability(Words) ) )   
