#Desafio 14 - Escreva um programa que leia uma temperatura em Celsius e a converta para Farenheit

c = float(input('Digite a temperatura em °C: '))
f = ((9*c)/5)+32
print('A temperatura de {}°C equivale a {}°F'.format(c, f))
