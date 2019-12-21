"""1. Função que imprima sequencioalmente"""


def imprimeSeq():
    n = int(input('Número de repetições: '))
    for i in range(n):
        print((str(i+1)+' ')*(i+1))

imprimeSeq()