#Desafio 34
#Escreva um programa que pergunte o salário de um funcionário e calcule o valor de seu aumento.

#Para salários superiores a R$ 1.250,00, calcule um aumento de 10%.
#Para salários inferiores ou iguais, o aumento é de 15%

sal = float(input('Digite o salário do funcionário: '))

if sal > 1250:
    aum = (sal*10/100)
    print('Será dado um aumento de 10% no valor de R${:.2f}'.format(aum))
    print('O salário corrigido será de {:.2f}'.format(sal+aum))
else:
    aum = (sal*15/100)
    print('Será dado um aumento de 15% no valor de R${:.2f}'.format(aum))
    print('O salário corrigido será de R${:.2f}'.format(sal+aum))