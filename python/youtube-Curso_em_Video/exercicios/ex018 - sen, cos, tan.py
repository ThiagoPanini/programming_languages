#Desafio 18
#Faça um programa que leia um ângulo qualquer e mostre na tela o valor do seno, cosseno e tangente desse ângulo

import math

angulo = float(input('Digite um ângulo: '))
rad = angulo*3.141592653589793/180 #Macete: transformar o ângulo em radianos. Verificar se há o número 'Pi' pronto
sen = math.sin(rad)
cos = math.cos(rad)
tan = math.tan(rad)
print('O seno do angulo {} é igual a {:.3f}'.format(angulo, sen))
print('O cosseno do angulo {} é igual a {:.3f}'.format(angulo, cos))
print('A tangente do do angulo {} é igual a {:.3f}'.format(angulo, tan))