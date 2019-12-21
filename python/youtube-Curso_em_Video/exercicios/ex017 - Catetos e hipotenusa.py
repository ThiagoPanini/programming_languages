#Desafio 17
#Faça um programa que leia o comprimento do cateto oposto e do cateto adjacente de um triângulo retângulo
#e mostre o comprimento da hipotenusa

from math import sqrt

co = float(input('Digite o cateto oposto do triângulo: '))
ca = float(input('Digite o cateto adjacente do triângulo: '))
h = sqrt((pow(co,2))+(pow(ca,2)))

print('A hipotenusa vale {}'.format(h))
