#Desafio 19
#Um professor quer sortear um de seus quatro alunos para apagar o quadro.
#Faça um programa que ajude ele, lendo o nome deles e escrevendo o nome do escolhido.

from random import randint, choice

a1 = input('Nome do aluno número 1: ')
a2 = input('Nome do aluno número 2: ')
a3 = input('Nome do aluno número 3: ')
a4 = input('Nome do aluno número 4: ')

lista = [a1, a2, a3, a4] #Criar uma lista de alunos - atenção para os colchetes

sorteio = randint(1,4)


print('O aluno escolhido é o de número {}'.format(sorteio))
#print('O aluno escolhido é: {}'.format(lista[sorteio-1]))

#__________________ Outra forma de fazer este exercício______________________

#Após a criação da lista...

escolhido = choice(lista) #Escolhe um valor aleatório da lista
print('O aluno escolhido foi {}'.format(escolhido))