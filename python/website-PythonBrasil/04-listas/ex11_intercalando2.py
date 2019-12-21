"""4.11 Altere o programa anterior, intercalando 3 vetores de 10 elementos cada."""

lista1 = [int(input(f'Insira o elemento {x+1} da Lista 1: ')) for x in range(10)]
print('\n')
lista2 = [int(input(f'Insira o elemento {i+1} da Lista 2: ')) for i in range(10)]
print('\n')
lista3 = [int(input(f'Insira o elemento {c+1} da Lista 2: ')) for c in range(10)]
print('\n')

print(f'Lista 1: {lista1}')
print(f'Lista 2: {lista2}')
print(f'Lista 3: {lista3}')

lista4 = []
for item in range(10):
    lista4.append(lista1[item])
    lista4.append(lista2[item])
    lista4.append(lista3[item])
print(f'Lista 3: {lista4}')