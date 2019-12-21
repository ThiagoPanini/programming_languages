# Desafio 52
# Faça um programa que leia um número inteiro e diga se ele é ou não um número primo

n = int(input('Digite um número para verificar se é PRIMO: '))
"""c = 2
b = 0
for c in range(c, int((n/2)+1)):
    if n % c == 0:
        print('{} é divisível por {}. Não é primo.'.format(n, c))
        b = 1
if b == 0:
    print('{} é divisível apenas por 1 e por ele mesmo. É primo.'.format(n))
print('Fim')"""
# Modo Guanabara
cont = 0
for c in range(1, n + 1):
    if n % c == 0:
        print('\033[4;34m', c, '\033[m' , end=' ')
        cont += 1
    else:
        print('\033[30m', c, end=' ')
if cont > 2:
    print('\nO número {} é divisível {} vezes. Não é primo.'.format(n, cont))
else:
    print('\nO número {} é PRIMO.'.format(n))
