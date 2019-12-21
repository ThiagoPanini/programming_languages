#Desafio 16
#Crie um programa que leia um número real qualquer pelo teclado e mostre na tela a sua porção inteira
#Exemplo: Digite o número 6.127. O número 6.127 tem a parte inteira 6

import math

num = float(input('Digite um número real qualquer: '))
print('O número {} possui parte inteira {}'.format(num, math.floor(num)))