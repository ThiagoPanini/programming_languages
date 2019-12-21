#!/usr/bin/env python
import sys
import re

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+|[.,!?;]"

if __name__ == '__main__':
    fileName = sys.argv[1]
    
    document = open(fileName,'r')
    content  = document.read()

    for (i, w) in enumerate( re.findall(regex, content) ):
        entity = "" # Nao-importante"
        if w[0].isupper():
            entity = "<- IMPORTANTE"
        print ("{} {}  {}".format(i, w, entity))

