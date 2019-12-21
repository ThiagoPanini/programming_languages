# Desafio 60
# Faça um programa que leia um número qualquer e mostre seu fatorial.
# Ex: 5! = 5x4x3x2x1 = 120
from time import sleep
from math import factorial

n = int(input('Digite um número inteiro para verificar seu FATORIAL: '))
i = n
fatorial = n
rep = '{}! = {}'.format(n, fatorial)
print('='*n*6)
while i > 1:
    fatorial = fatorial * (i-1)
    rep = rep + ' x {}'.format(i-1)
    print('{} = {}'.format(rep, fatorial))
    sleep(0.75)
    i = i-1
print('='*n*6)
print('\033[1;30m{}! é igual a: {}\033[m'.format(n, fatorial))
sleep(1)
print('\n')
print('-'*50)
print('MODO 2: FATORIAL EM UMA ÚNICA LINHA (RESOLUÇÃO GUANABARA)')
print('-'*50)
n1 = int(input('Digite um número inteiro para verificar seu FATORIAL: '))
c = n
f = 1 # Fator nulo da multiplicação (Se fosse soma, esta variável teria de ser 0)
print('Calculando {}! = '.format(n), end='') #Inicia a apresentação em uma linha (end='' "concatena" o próximo print)
while c > 0:
    print('{}'.format(c), end='') # Printa os números todos na mesma linha e separados por um espaço (end='')
    print(' x ' if c > 1 else ' = ', end='') #Para não printarmos um x até no último número, utilizamos este comando.
    # Acima vemos uma estrutura de if em linha responsávle por verificar se 'c' está abaixo de 1 (último número)
    # Enquanto c estiver acima de 1 (contagem descrecente), escrevemos 'x', caso contrário (c = 0), escrevemos '='
    f *= c # Mesma coisa que f = f * c -> variável f vai acumulando multiplicações sucessivas
    c -= 1
print('{}'.format(f))
print('\n')
sleep(1)
print('-'*50)
print('MODO 3: MODO FACIL -> BIBLIOTECA MATH (FACTORIAL)')
print('-'*50)
n2 = int(input('Digite um número inteiro para verificar seu FATORIAL: '))
fact = factorial(n)
print('{}! é igual a {}'.format(n, fact))

