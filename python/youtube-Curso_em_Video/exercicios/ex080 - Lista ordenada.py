"""Exercício 80
Crie um programa onde o usuário possa digitar cinco valores numéricos e cadastre-os em uma lista
já na posição correta de inserção (sem utilizar o sort()). No final, mostre a lista ordenada na tela
"""

numeros = []
for i in range(5):
    n = int(input('Digite um valor: '))
    if i == 0:
        numeros.append(n)
        print(f'Número {n} adicionado no final da lista.')
    else:
        if n == max(numeros) or n > max(numeros):
            print(f'Número {n} adicionado no final da lista.')
            numeros.append(n)
        elif n == min(numeros) or n < min(numeros):
            print(f'Número {n} adicionado no início da lista.')
            numeros.insert(0, n)
        else:
            for num in numeros:
                if n < num:
                    print(f'Número {n} adicionado na posição {numeros.index(num)} da lista.')
                    numeros.insert(numeros.index(num), n)
                    break

print('-'*15)
print('{:^15}'.format('LISTA INSERIDA'))
print('-'*15)
print(numeros)
