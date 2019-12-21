"""1.15 Faça um Programa que pergunte quanto você ganha por hora e o número de horas
trabalhadas no mês. Calcule e mostre o total do seu salário no referido mês,
sabendo-se que são descontados 11% para o Imposto de Renda, 8% para o INSS e
5% para o sindicato, faça um programa que nos dê:
 a. salário bruto.
 b. quanto pagou ao INSS.
 c. quanto pagou ao sindicato.
 d. o salário líquido.
 e. calcule os descontos e o salário líquido, conforme a tabela
abaixo:

+ Salário Bruto : R$
- IR (11%) : R$
- INSS (8%) : R$
- Sindicato ( 5%) : R$
= Salário Liquido : R$

 Obs.: Salário Bruto - Descontos = Salário Líquido."""

valorhora = float(input('Quanto ganha por hora? R$'))
horames = int(input('Quantas horas trabalha por mês? '))
salarioliquido = valorhora * horames
ir = salarioliquido*11/100
inss = salarioliquido*8/100
sindicato = salarioliquido*5/100
salariobruto = salarioliquido-ir-inss-sindicato

print(f'Salário bruto: R${salariobruto:.2f}.')
print(f'IR (11%): R${ir:.2f}.')
print(f'INSS (8%): R${inss:.2f}.')
print(f'Sindicato (5%): R${sindicato:.2f}.')
print(f'Salário líquido: R${salarioliquido:.2f}.')

