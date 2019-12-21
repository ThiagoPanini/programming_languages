#Importando Módulos e trabalhando com as Bibliotecas

#Biblioteca math

import math

num = int(input('Digite um número: '))
raiz = math.sqrt(num)
print('A raíz de {} é igual a {:.2f}'.format(num, raiz)) #uma forma de arredondar com 2 casas decimais
print('A raíz de {} é igual a {}'.format(num, math.ceil(raiz))) #arredonda para cima
print('A raíz de {} é igual a {}'.format(num, math.floor(raiz))) #arredonda para baixo

