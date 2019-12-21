"""2.18 Faça um programa que calcule as raízes de uma equação do segundo grau, na forma ax2 + bx + c.
O programa deverá pedir os valores de a, b e c e fazer as consistências, informando ao usuário nas seguintes situações:
   a. Se o usuário informar o valor de A igual a zero, a equação não é do segundo grau e o
   programa não deve fazer pedir os demais valores, sendo encerrado;
   b. Se o delta calculado for negativo, a equação não possui raizes reais. Informe ao usuário e encerre o programa;
   c. Se o delta calculado for igual a zero a equação possui apenas uma raiz real; informe-a ao usuário;
   d. Se o delta for positivo, a equação possui duas raiz reais; informe-as ao usuário; """

from math import sqrt

while True:
    print('Insira os valores de "a", "b" e "c" de acordo com a equação do segundo grau "ax² + bx + c"')
    a = float(input('a: '))
    b = float(input('b: '))
    c = float(input('c: '))
    if a == 0:
        print('Valor de "a" igual a 0 -> Equação de primeiro grau. Encerrando programa.')
        break
    delta = (b**2) - 4*a*c
    if delta == 0:
        x = -b / 2*a
        print(f'Raíz única de valor {x:.2f}!')
    elif delta < 0:
        print('Raízes negativas. Encerrando programa.')
        break
    elif delta > 0:
        x1 = (-b + sqrt(delta)) / 2*a
        x2 = (-b - sqrt(delta)) / 2*a
        print(f'Raízes duplas de valores {x1:.2f} e {x2:.2f}')
    break
print('Fim')