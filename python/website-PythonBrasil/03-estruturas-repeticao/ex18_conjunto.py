"""3.18 Faça um programa que, dado um conjunto de N núms, determine o menor, o maior valor e a soma dos valores."""

conjunto = []
elementos = int(input('Quantos elementos terá o conjunto? '))
maior = menor = 0

for c in range(elementos):
    n = float(input(f'Digite o {c+1}º do conjunto: '))
    conjunto.append(n)
print(f'Nosso conjunto de valores: {conjunto}')

for item in conjunto:
    if item == conjunto[0]:
        maior = menor = item
    else:
        if item > maior:
            maior = item
        elif item < menor:
            menor = item
print(f'Maior valor: {maior}\nMenor valor: {menor}')