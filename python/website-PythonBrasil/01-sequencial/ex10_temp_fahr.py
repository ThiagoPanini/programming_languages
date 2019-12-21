"""1.10 Faça um Programa que peça a temperatura em graus Celsius, transforme e mostre
em graus Farenheit."""

c = float(input('Digite a temperatura em graus °C: '))

fahr = ((9*c)/5)+32

print(f'A temperatura de {c:.2f} °C equivale a {fahr:.2f} °F.')