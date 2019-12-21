"""3.19 Altere o programa anterior para que ele aceite apenas números entre 0 e 1000."""

conjunto = []
elementos = int(input('Quantos elementos terá o conjunto? '))
maior = menor = 0

for c in range(elementos):
    n = float(input(f'Digite o {c+1}º do conjunto (entre 0 e 1000): '))
    while n not in range(1001):
        n = float(input('Número inválido. Digite entre 0 e 1000: '))
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