#Desafio 13 - Faça um algorítmo que leia o salário de um funcionário e mostre seu novo salário com 15% de aumento

s = float(input('Digite o salário do funcionário: '))
aum = float(input('Insira a porcentagem de aumento a ser aplicada: '))

real = s*(aum/100)

print('Novo salário com aumento de {}% é igual a R$ {:.2f}'.format(aum, s+real))