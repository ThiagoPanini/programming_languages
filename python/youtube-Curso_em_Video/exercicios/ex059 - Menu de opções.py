# Desafio 59
# Crie um programa que leia dois valores e mostre um menu na tela:
# [1] somar
# [2] multiplicar
# [3] maior
# [4] novos números
# [5] sair do programa
# Seu programa deverá realizar a operação solicitada em cada caso
import emoji
from time import sleep
opcoes = [1, 2, 3, 4, 5]
opt = 0
n1 = int(input('Primeiro número inteiro: '))
n2 = int(input('Segundo número inteiro: '))

while opt != 5:
    print('='*30)
    print('ESCOLHA UMA DAS OPÇÕES ABAIXO')
    print('='*30)
    print('[ 1 ] Somar', emoji.emojize(':heavy_plus_sign:', use_aliases=True),
        '\n[ 2 ] Multiplicar', emoji.emojize(':heavy_multiplication_x:', use_aliases=True),
        '\n[ 3 ] Maior', emoji.emojize(':point_up:',use_aliases=True),
        '\n[ 4 ] Novos números', emoji.emojize(':repeat:',use_aliases=True),
        '\n[ 5 ] Sair do programa', emoji.emojize(':no_entry_sign:',use_aliases=True))
    opt = int(input('Digite: '))

    if opt == 1:
        print('A SOMA entre {} e {} é igual a {}.'.format(n1, n2, n1+n2))
    if opt == 2:
        print('A MULTIPLICAÇÃO entre {} e {} é igual a {}'.format(n1, n2, n1*n2))
    if opt == 3:
        if n1 > n2:
            print('O MAIOR número é {}.'.format(n1))
        elif n2 > n1:
            print('O MAIOR número é {}'.format(n2))
        else:
            print('Os números são IGUAIS!')
    if opt == 4:
        print('Informe os NOVOS valores:')
        n1 = int(input('Primeiro número inteiro: '))
        n2 = int(input('Segundo número inteiro: '))
    if opt not in opcoes:
        print('Opção INVÁLIDA. Digite novamente.')
    sleep(0.75)
print('\nFim do programa.')
