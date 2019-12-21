# Desafio 63
# Escreva um programa que leia um número n inteiro qualquer e mostre na tela os n primeiros elementos
# de uma Sequência de Fibonacci.
# Ex: 0 -> 1 -> 1 -> 2 -> 3 -> 5 -> 8

from time import sleep
n = int(input('Até qual termo deseja ver a sequência de FIBONACCI? '))
i = 1
anterior = anterior1 = 0
atual = atual1 = 1
proximo = proximo1 = anterior + atual
seq = seq1 = '{}'.format(anterior)
print('-'*50)
print('MOSTRANDO SEQUENCIA DE FIBONACCI ATÉ O {}º TERMO'.format(n))
print('-'*50)
while i <= n:
    print('Linha {:2}: {}'.format(i, seq))
    sleep(0.25)
    seq = seq + '→{}'.format(proximo)
    proximo = atual + anterior
    anterior = atual
    atual = proximo
    i += 1
print('FIM')

# RESOLUÇÃO GUANABARA

sleep(0.5)
print('-'*45)
print('MOSTRANDO SEQUENCIA EM UMA ÚNICA LINHA'.format(n))
print('-'*45)
sleep(0.5)
t1 = 0
t2 = 1
print('{}→{}'.format(t1, t2), end='')
i = 3   # Contador já começa no 3 pois eu já mostrei os 2 primeiros termos (0 e 1)
while i <= n:
    t3 = t1+ t2
    print('→{}'.format(t3), end='')
    t1 = t2
    t2 = t3
    i += 1
print('→FIM')
