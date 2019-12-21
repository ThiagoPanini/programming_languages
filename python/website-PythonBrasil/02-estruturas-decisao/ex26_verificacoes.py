"""2.26 Faça um Programa que leia 2 números e em seguida pergunte ao usuário qual operação ele deseja realizar.
O resultado da operação deve ser acompanhado de uma frase que diga se o número é:
a. par ou ímpar;
b. positivo ou negativo;
c. inteiro ou decimal."""

from math import trunc

n1 = float(input('Primeiro número: '))
n2 = float(input('Segundo número: '))

escolhas = ('soma', 'subtração', 'multiplicação', 'divisão')

choice = int(input('Escolha uma operação: \n'
                   '[ 1 ] SOMA (a + b)\n'
                   '[ 2 ] SUBTRAÇÃO (a - b)\n'
                   '[ 3 ] MULTIPLICAÇÃO (a * b)\n'
                   '[ 4 ] DIVISÃO (a / b)\n'))

while choice not in range(1, 5):
    print('Valor inválido.')
    choice = int(input('Escolha uma operação: \n'
                       '[ 1 ] SOMA (a + b)\n'
                       '[ 2 ] SUBTRAÇÃO (a - b)\n'
                       '[ 3 ] MULTIPLICAÇÃO (a * b)\n'
                       '[ 4 ] DIVISÃO (a / b)\n'))

if choice == 1:
    result = n1 + n2
elif choice == 2:
    result = n1 - n2
elif choice == 3:
    result = n1 * n2
elif choice == 4:
    result = n1 / n2
print(f'O resultado da operação {escolhas[choice-1]} entre {n1} e {n2} é igual a {result:.2f}. ')
if result % 2 == 0:
    print('Número PAR.')
else:
    print('Número ÍMPAR.')
if result < 0:
    print('Número NEGATIVO.')
else:
    print('Número POSITIVO.')
if result / trunc(result) == 1:
    print('Número INTEIRO.')
else:
    print('Número DECIMAL.')


