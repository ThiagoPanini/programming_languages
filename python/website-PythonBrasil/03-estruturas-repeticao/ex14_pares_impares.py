"""3.14 Faça um programa que peça 10 números inteiros, calcule e mostre a quantidade de números pares e a quantidade de
 números impares. """

pares = impares = 0

for c in range(10):
    n = int(input(f'Digite o {c+1}º número: '))
    if n % 2 == 0:
        pares += 1
    else:
        impares += 1
print(f'Quantidade de números pares: {pares}.\nQuantidade de números ímpares {impares}.')
