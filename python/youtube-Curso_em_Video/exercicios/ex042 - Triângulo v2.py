#Desafio 42
#Refaça o Desafio 035 dos triângulos acrescentando o recurso de mostrar que tipo de triângulo será formado:
# - Equilátero: todos os lados iguais.
# - Isósceles: dois lados iguais.
# - Escaleno: todos os lados diferentes.

import time
from math import acos

r1 = float(input('Insira o comprimento da reta 1 do triângulo: '))
r2 = float(input('Insira o comprimento da reta 2 do triângulo: '))
r3 = float(input('Insira o comprimento da reta 3 do triângulo: '))

if r1+r2 < r3 or r1+r3 < r2 or r2+r3 < r1:
    print('\nNão é possível formar um triângulo com os lados inseridos.'
          'A soma dos dois MENORES lados não equivale ao MAIOR')
else:
    print('\nÉ possível formar um triângulo com os lados inseridos. '
          '\nOs lados são {}, {} e {}. Verificando tipo do triângulo...'.format(r1, r2, r3))
    time.sleep(3)
    if r1 == r2 == r3:
        print('Triângulo EQUILÁTERO -> Todos os lados são iguais!')
    elif r1 != r2 != r3:
        print('Triângulo ESCALENO -> Todos os lados são diferentes!')
    else:
        print('Triângulo ISÓSCELES -> Dois lados iguais e um diferente!')

    print('\nCalculando os ângulos...')
    time.sleep(3) #Delay para pensar...
    rada1 = acos(((r1 ** 2) - (r2 ** 2) - (r3 ** 2)) / (-2 * r2 * r3)) # Lei dos cossenos
    a1 = (rada1 * 180 / 3.1415) # Transformação em radianos para apresentar na variável
    print('O ângulo alfa é igual a {:.2f}°'.format(a1))

    rada2 = acos(((r2 ** 2) - (r1 ** 2) - (r3 ** 2)) / (-2 * r1 * r3))
    a2 = (rada2 * 180 / 3.1415)
    print('O ângulo beta é igual a {:.2f}°'.format(a2))

    rada3 = acos(((r3 ** 2) - (r1 ** 2) - (r2 ** 2)) / (-2 * r1 * r2))
    a3 = (rada3 * 180 / 3.1415)
    print('O ângulo gama é igual a {:.2f}°'.format(a3))

    if int(a1) == 90 or int(a2) == 90 or int(a3) == 90:
        print('Temos um ângulo reto. Triângulo RETÂNGULO!')

    print('\nA somatória dos ângulos é igual a {:.0f}°'.format(a1 + a2 + a3))

    print('\nLINK PARA VERIFICAÇÃO DOS CÁLCULOS: https://es.planetcalc.com/534/?language_select=es')
