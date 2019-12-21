"""4.10 Faça um Programa que leia dois vetores com 10 elementos cada. Gere um terceiro vetor de 20 elementos,
cujos valores deverão ser compostos pelos elementos intercalados dos dois outros vetores."""

lista1 = [int(input(f'Insira o elemento {x+1} da Lista 1: ')) for x in range(10)]
print('\n')
lista2 = [int(input(f'Insira o elemento {i+1} da Lista 2: ')) for i in range(10)]
print('\n')

print(f'Lista 1: {lista1}')
print(f'Lista 2: {lista2}')

lista3 = []
for item in range(10):
    lista3.append(lista1[item])
    lista3.append(lista2[item])

print(f'Lista 3: {lista3}')