"""3.8 Faça um programa que leia 5 números e informe a soma e a média dos números."""

print('='*28)
print('Soma e média entre 5 números')
print('='*28)

soma = 0

for c in range(1, 6):
    num = float(input(f'Digite o {c}º número: '))
    soma += num
media = soma/5

print(f'A soma entre os valores é {soma:.2f} e a média é {media:.2f}')