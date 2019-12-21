#!/usr/bin/env python
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
    
        for f in sys.argv[1:]:
            infile = open(f, 'r')
            
            while 1:
                output = ""
                word   = ""
                line   = infile.readline()
                
                if line == "":
                    break
                for c in line:
                    if c.isalpha():
                        word += c.lower()
                    else:
                        if word:
                            output += word[0:3]
                            word = ""
                        output += c.lower()
                print (output, end='')
            infile.close()

