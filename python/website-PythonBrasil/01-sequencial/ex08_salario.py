"""1.8 Faça um Programa que pergunte quanto você ganha por hora e o número de horas
trabalhadas no mês. Calcule e mostre o total do seu salário no referido mês."""

porhora = float(input('Quanto você ganha por hora? R$'))
h = int(input('Quantas horas trabalhadas por dia? '))
print('\nConsiderando 22 dias úteis de trabalho em um mês...')
print(f'Seu salário é igual a R${porhora*h*22}.')
