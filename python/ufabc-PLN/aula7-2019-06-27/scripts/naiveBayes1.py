import sys
import re
import math
regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+" 

class NBClassifier:

    def __init__(self, training_file=None):
        self.Data          = []
        self.Classes       = dict([])
        self.V             = set([])
        self.bigdoc        = dict([])

        self.logprior      = dict([])
        self.loglikelihood = dict([])

        if training_file is not None:
            self.load_data(training_file)


    def load_data(self, training_file):
        training_document = open(training_file,'r')

        for line in training_document.readlines():
            c, d = tuple(line.strip().split("\t"))
            self.Data.append((c,d))

            if c not in self.Classes:
                self.Classes[c] = 0
                self.bigdoc[c] = []
            self.Classes[c] += 1
            
            for w in re.findall(regex, d):
                self.V.add(w)
                self.bigdoc[c].append(w)

        print("Total: classes={} documentos={} vocabulario={}".format(len(self.Classes), len(self.Data), len(self.V) ) )
        print("\n", self.V)
        print("\n", self.bigdoc)


    def train(self):
        for c in self.Classes:
            Ndoc = len(self.Data)
            Nc   = self.Classes[c]

            self.logprior[c]  = Nc/Ndoc

            count_wc = 0
            for w in self.V:
                count_wc += self.bigdoc[c].count(w)

            for w in self.V:
                self.loglikelihood[(w,c)] = (self.bigdoc[c].count(w)+1)/(count_wc + len(self.V))  

        print("\n", self.logprior)
        print("\n", self.loglikelihood)


    def test(self, testdoc):
        # TODO
        print ( "Testando: {}".format(testdoc) )

        return list(self.Classes.keys())[0]

        
if __name__ == '__main__':
    fileName = sys.argv[1]

    NBC = NBClassifier(fileName)
    NBC.train()

    print("\n Teste 1: {}".format( NBC.test("predictable with no fun")) )
