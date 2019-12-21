#Importando Módulos e trabalhando com as Bibliotecas

#Biblioteca random

import random

num = random.random() #Gera um número aleatório entre 0 e 1
print(num)

num2 = random.randint(1,10) #Gera um número aleatório entre 1 e 10
print(num2)

lista = ['a', 'B', 'C']
print(lista)
lista[1] = '\033[1m' + 'b' + '\033[m'
print(lista[1].replace('\033[1m',''))
print(lista[1].replace('\033[m',''))
print(lista)
print('\033[1;34mTeste')

