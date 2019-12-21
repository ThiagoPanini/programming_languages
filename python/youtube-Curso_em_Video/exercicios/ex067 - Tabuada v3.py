# Desafio 67
# Faça um programa que mostre a tabuada de vários números, um de cada vez, para cada valor digitado pelo usuário.
# O programa será interrompido quando o número solicitado for negativo.

while True:
    print('-'*40)
    n = int(input('Digite um número para ver sua tabuada: '))
    print('-'*40)
    if n < 0:
        print('~'*20)
        print('PROGRAMA ENCERRADO.')
        print('~'*20)
        break
    i = 1
    while i <= 10:
        print(f'{n} x {i:2} = {n*i}')
        i += 1
