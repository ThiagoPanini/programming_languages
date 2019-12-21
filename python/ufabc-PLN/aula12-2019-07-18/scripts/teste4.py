# coding=utf8
import sys
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as img

M = 1000

def hashF(word):
    k = 0
    for i in range(0, len(word)):
        k += (17**i) * ord(word[i])
    return k%M

if __name__ == '__main__':
    dataset = sys.argv[1]
    
    Mcollision = numpy.zeros(M)

    for s in open(dataset,'r').readlines():
        index = hashF(s.strip().lower())
        Mcollision[index] += 1


    plt.bar(range(M), Mcollision, 1, color="blue")
    plt.show()


