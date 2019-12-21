#!/usr/bin/env python
import sys
import re

regex = r"[-'a-zA-ZÀ-ÖØ-öø-ÿ]+|[.,!?;]"

if __name__ == '__main__':
    fileName = sys.argv[1]
    
    document = open(fileName,'r')
    content  = document.read()
    words    = re.findall(regex, content)

    for (i, w) in enumerate(words):
        if w[0].isupper() and i>=1 and words[i-1] not in ".,!?;":
            print ("{} {}  <- IMPORTANTE".format(i, w))
        else:
            print ("{} {} ".format(i, w))

