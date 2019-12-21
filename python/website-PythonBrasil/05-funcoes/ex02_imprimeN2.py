"""2. Imprime números"""

def imprimeSeq():
    """Imprime sequência de números de acordo com o número de repetições inserida"""
    n = int(input('Número de repetições: '))
    for i in range(n):
        print(i+1)
        for j in range(i+1):
            print(j+1, end=' ')


imprimeSeq()