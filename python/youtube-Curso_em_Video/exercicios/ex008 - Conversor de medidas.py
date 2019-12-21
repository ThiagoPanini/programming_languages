#Desafio 8 - Escreva um programa que leia o valor em metro e o converta em centímetros e milímetros

m = float(input('Insira o valor em metros: '))
print('Valor em centímetros: {} cm\nValor em milímetros: {} mm'.format(m/100, m/1000))