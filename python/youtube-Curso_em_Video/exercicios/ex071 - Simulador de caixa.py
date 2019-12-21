""" Desafio 71
Crie um programa que simule o funcionamento de um caixa eletrônico.
No início, pergunte ao usuário qual será o valor a ser sacado (número inteiro) e o programa vai informar
quantas cédulas de cada valor serão entregues.
Obs: Considere que o caixa possui cédulas de R$50, R$20, R$10 e R$1"""

from time import sleep

ced50 = ced20 = ced10 = ced1 = 0

cores = ['\033[1;30m', '\033[1;33m', '\033[1;34m', '\033[m', '\033[1;35m', '\033[1;32m', '\033[1;31m']
# cores[0] = branco; cores[1] = amarelo; cores[2] = azul; cores[3] = padrão; cores[4] = roxo; cores[5] = verde;
# cores[6] = vermelho

print('{}='.format(cores[0])*30)
print(cores[1], '{:~^50}'.format(cores[2] + ' PYTHON ' + cores[1] + 'BANK ' + cores[2]))
print('{}='.format(cores[0])*30)
print(cores[3])

cedulas = [50, 20, 10, 1]
c = nota = 0
valor = int(input('Digite o valor a ser sacado: R$'))

while valor != 0:
    if (valor/cedulas[c]) >= 1:
        nota = valor // cedulas[c]
        valor -= nota * cedulas[c]
        print(f'Você receberá {nota} notas de R${cedulas[c]}')
        c += 1
    else:
        c += 1



"""print('Você receberá: ')
if (valor/50) >= 1:
    ced50 = valor // 50
    valor -= (ced50*50)
    print(f'{ced50} Cédulas de {cores[0]}R$50{cores[3]}')
    sleep(1)
if (valor/20) >= 1:
    ced20 = valor // 20
    valor -= (ced20*20)
    print(f'{ced20} Cédulas de {cores[1]}R$20{cores[3]}')
    sleep(1)
if (valor/10) >= 1:
    ced10 = valor // 10
    valor -= (ced10*10)
    print(f'{ced10} Cédulas de {cores[6]}R$10{cores[3]}')
    sleep(1)
if (valor/5) >= 1:
    ced5 = valor // 5
    valor -= (ced5*5)
    print(f'{ced5} Cédulas de {cores[4]}R$5{cores[3]}')
    sleep(1)
if valor != 0:
    ced1 = valor
    valor -= ced1
    print(f'{ced1} Cédulas de {cores[5]}R$1{cores[3]}')
    sleep(1)"""

print('\nVolte sempre ao PYTHON BANK! Tenha um bom dia!')