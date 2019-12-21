"""1.9 Faça um Programa que peça a temperatura em graus Farenheit, transforme e
mostre a temperatura em graus Celsius.
 C = (5 * (F-32) / 9)."""

f = float(input('Insira a temperatura em °F: '))
celsius = (5*(f-32)/9)
print(f'A temperatura de {f:.2f} °F convertida em Celsius é igual a {celsius:.2f} °C.')