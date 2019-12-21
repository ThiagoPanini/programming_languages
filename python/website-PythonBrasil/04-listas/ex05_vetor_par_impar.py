"""4.5 Faça um Programa que leia 20 números inteiros e armazene-os num vetor.
Armazene os números pares no vetor PAR e os números IMPARES no vetor impar. Imprima os três vetores."""

par = []
impar = []
for i in range(20):
    item = int(input('Digite o {}º elemento: '.format(i+1)))
    if item % 2 == 0:
        par.append(item)
    else:
        impar.append(item)
print(f'Lista PAR: {par}\nLista ÍMPAR: {impar}')