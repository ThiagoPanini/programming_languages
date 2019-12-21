#Deafio 35
#Desenvolva um programa que leia o comprimento de três retas e diga ao usuário se elas podem ou não formar um triângulo

from math import acos, cos

r1 = float(input('Insira o comprimento da reta 1: '))
r2 = float(input('Insira o comprimento da reta 2: '))
r3 = float(input('Insira o comprimento da reta 3: '))
print('\n')

"""Para verificar se as retas formam um triângulo, 
é necessário saber se a soma de duas delas é maior ou igual o comprimento da terceira,
ou seja, precisamos identificar a maior reta entre as três e verificar se a soma das outras duas é maior ou igual
ao comprimento desta reta maior."""

if r1 > r2 and r1 > r3:
    if r2 + r3 >= r1:
        print('É possível formar um triângulo com as retas {}, {} e {}'.format(r1, r2, r3))
        print('Isto pois {} + {} >= {} (Maior reta)'.format(r2, r3, r1))
        tri = True
    else:
        print('Não é possível formar um triângulo com as retas {}, {} e {}'.format(r1, r2, r3))
        print('Isto pois {} + {} é menor do que {} (Maior reta)'.format(r2, r3, r1))
        tri = False
if r2 > r1 and r2 > r3:
    if r1 + r3 >= r2:
        print('É possível formar um triângulo com as retas {}, {} e {}'.format(r2, r1, r3))
        print('Isto pois {} + {} >= {} (Maior reta)'.format(r1, r3, r2))
        tri = True
    else:
        print('Não é possível formar um triângulo com as retas {}, {} e {}'.format(r2, r1, r3))
        print('Isto pois {} + {} é menor do que {} (Maior reta)'.format(r1, r3, r2))
        tri = False
if r3 > r1 and r3 > r2:
    if r1 + r2 >= r3:
        print('É possível formar um triângulo com as retas {}, {} e {}'.format(r3, r1, r2))
        print('Isto pois {} + {} >= {} (Maior reta)'.format(r1, r2, r3))
        tri = True
    else:
        print('Não é possível formar um triângulo com as retas {}, {} e {}'.format(r3, r1, r2))
        print('Isto pois {} + {} é menor do que {} (Maior reta)'.format(r1, r2, r3))
        tri = False

print('\n')

if tri == True:
    rada1 = acos(((r1**2)-(r2**2)-(r3**2))/(-2*r2*r3))
    a1 = (rada1*180/3.1415)
    print('O ângulo alfa é igual a {:.2f}°'.format(a1))

    rada2 = acos(((r2**2)-(r1**2)-(r3**2))/(-2*r1*r3))
    a2 = (rada2*180/3.1415)
    print('O ângulo beta é igual a {:.2f}°'.format(a2))

    rada3 = acos(((r3**2)-(r1**2)-(r2**2))/(-2*r1*r2))
    a3 = (rada3*180/3.1415)
    print('O ângulo gama é igual a {:.2f}°'.format(a3))

    print('A somatória dos ângulos é igual a {:.2f}°'.format(a1+a2+a3))

    print('\nLINK PARA VERIFICAÇÃO DOS CÁLCULOS: https://es.planetcalc.com/534/?language_select=es')
else:
    print('Como não há um triângulo, não há como calcular os ângulos.')



