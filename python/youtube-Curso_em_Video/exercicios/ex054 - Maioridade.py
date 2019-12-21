# Desafio 54
# Crie um programa que leia o ano de nascimento de sete pessoas.
# No final, mostre quantas pessoas ainda não atingiram a maioridade e quantas já são maiores

from datetime import date
i = 0
for c in range(0, 7):
    a = int(input('Insira o ano de nascimento (4 dígitos) da pessoa {}: '.format(c+1)))
    idade = date.today().year - a
    if idade >= 18:
        i = i + 1
print('{} pessoas são maiores de 18 anos e {} pessoas são menores de 18 anos.'.format(i, 7-i))