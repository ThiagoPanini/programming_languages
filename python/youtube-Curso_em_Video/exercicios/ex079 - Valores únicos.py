"""Exercício 79
Crie um programa onde o usuário possa digitar vários valores numéricos e cadastre-os em uma lista.
Caso o número já exista lá dentro, ele não será adicionado.
No final, serão exibidos os valores únicos digitados, em ordem crescente.
"""

r = ''
numeros = []
while True:
    num = int(input('Digite um número: '))
    if num not in numeros:
        numeros.append(num)
        print('Número adicionado com sucesso!')
    else:
        print('Não é possível adicionar números repetidos a lista.')
    r = input('Deseja continuar? [S/N] ').strip().upper()
    if r in 'N':
        break

print('-'*26)
print('{:^26}'.format('LISTA EM ORDEM CRESCENTE'))
print('-'*26)
print(sorted(numeros))