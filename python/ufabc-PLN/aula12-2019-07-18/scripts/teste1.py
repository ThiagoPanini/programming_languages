# coding=utf8
import sys
import re
import os


def hashF(word):
    k = 0
    for i in range(0, len(word)):
        k += ord(w[i])
    return k

if __name__ == '__main__':
    
    while True:
        w = input("\nDigite uma palavra: ") 
        print(" h = {}".format(hashF(w)) )

