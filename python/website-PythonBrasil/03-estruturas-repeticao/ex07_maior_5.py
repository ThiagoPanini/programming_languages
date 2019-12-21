"""3.7 Faça um programa que leia 5 números e informe o maior número."""

print('-'*22)
print('Maior entre 5 números')
print('-'*22)
maior = menor = 0

for c in range(1, 6):
    num = float(input(f'Digite o {c}º número: '))
    if c == 1:  # Se é a primeira verificação...
        maior = menor = num  # Ambos maior e menor recebem o número
    else:  # Se não é a primeira verificação...
        if num > maior:
            maior = num
        if num < menor:
            menor = num
print(f'O maior número foi {maior} e o menor foi {menor}.')

