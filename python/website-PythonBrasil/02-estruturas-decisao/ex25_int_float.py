"""2.25 Faça um Programa que peça um número e informe se o número é inteiro ou decimal.
Dica: utilize uma função de arredondamento. """

from math import ceil, trunc

n = float(input('Digite um número: '))
arred = trunc(n)

if n / arred == 1:
    print(f'O número {n:.0f} é INTEIRO.')
else:
    print(f'O número {n} é DECIMAL.')