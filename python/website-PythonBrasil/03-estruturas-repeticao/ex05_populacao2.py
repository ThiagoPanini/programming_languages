"""3.5 Altere o programa anterior permitindo ao usuário informar as populações e as taxas de crescimento iniciais.
Valide a entrada e permita repetir a operação. """

from datetime import date
from math import trunc

print('-'*10)
print('POPULAÇÃO 2')
print('-'*10)

inicialA = int(input('População inicial de A: '))
while inicialA < 0:
    inicialA = int(input('Número inválido. População inicial de A: '))
txA = float(input('Taxa anual de crescimento de A (%): '))/100
inicialB = int(input('População inicial de B: '))
while inicialB < 0:
    inicialB = int(input('Número inválido. População inicial de B: '))
txB = float(input('Taxa anual de crescimento de B (%): '))/100

ano = date.today().year
cont = tx = ''
year = 0
inicialA += inicialA*txA
inicialB += inicialB*txB

while cont in 'Ss':
    ano += year
    print(f'\nAo final de {ano}, teremos: \n'
          f'População A: {trunc(inicialA)} habitantes.\n'
          f'População B: {trunc(inicialB)} habitantes.\n')
    cont = input('Deseja continuar? [S/N] ').strip().upper()[0]
    while cont not in 'SN':
        cont = input('Valor inválido. Deseja continuar? [S/N] ').strip().upper()[0]
    if cont in 'Ss':
        tx = input('Deseja alterar as taxas anuais de crescimento? [S/N] ').strip().upper()[0]
        while tx not in 'SN':
            tx = input('Valor inválido. Deseja alterar as taxas anuais de crescimento? [S/N] ').strip().upper()[0]
        if tx in 'Ss':
            txA = float(input('Taxa anual de crescimento de A (%): ')) / 100
            txB = float(input('Taxa anual de crescimento de B (%): ')) / 100
        year = int(input('Deseja visualizar os números após quantos anos? '))
        while year < 0:
            year = int(input('Número inválido. Deseja visualizar os números após quantos anos? '))
        for i in range(0, year+1):
            inicialA += inicialA * txA
            inicialB += inicialB * txB
print('Fim.')






