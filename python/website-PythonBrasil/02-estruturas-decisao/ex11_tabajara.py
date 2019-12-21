"""2.11 As Organizações Tabajara resolveram dar um aumento de salário aos seus colaboradores
e lhe contraram para desenvolver o programa que calculará os reajustes.

2.12   Faça um programa que recebe o salário de um colaborador e o reajuste segundo o seguinte critério,
baseado no salário atual:
   * salários até R$ 280,00 (incluindo)  : aumento de 20%
   * salários entre R$ 280,00 e R$ 700,00 : aumento de 15%
   * salários entre R$ 700,00 e R$ 1500,00 : aumento de 10%
   * salários de R$ 1500,00 em diante   : aumento de 5%

   Após o aumento ser realizado, informe na tela:
   a. o salário antes do reajuste;
   b. o percentual de aumento aplicado;
   c. o valor do aumento;
   d. o novo salário, após o aumento."""

salario = float(input('Digite o salário: R$'))
perc = 0
print(f'Salário antes do reajuste: R${salario:.2f}')

if salario <= 280:
    perc = 20/100
    salario += salario*perc
elif 280 < salario <= 700:
    perc = 15/100
    salario += salario*perc
elif 700 < salario <= 1500:
    perc = 10/100
    salario += salario * perc
elif salario > 1500:
    perc = 5/100
    salario += salario * perc

print(f'Porcentagem aplicada: {perc*100}%')
print(f'Salário corrigido: R${salario:.2f}')