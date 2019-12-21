"""1.6 Faça um Programa que peça o raio de um círculo, calcule e mostre sua área."""

from math import pi

r = float(input('Insira o raio da circunferência: '))

area = pi*(r**2)

print(f'A área de uma circunferência de raio {r} é igual a {area:.2f}.')