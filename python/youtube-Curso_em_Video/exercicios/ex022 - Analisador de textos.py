#Desafio 22
#Crie um programa que leia o nome completo de uma pessoa e mostre:
# - O nome com todas as letras maiúsculas;
# - O nome com todas minúsculas;
# - Quantas letras ao todo (sem considerar espaço)
# - Quantas letras tem o primeiro nome

nome = input('Digite seu nome completo: ')

print('Nome com todas maiúsculas: {}'.format(nome.upper())) #Transforma tudo em maiúscula
print('Nome com todas as minúsculas: {}\n'.format(nome.lower())) #Transforma tudo em minúscula

print('Contagem total de caracteres: {}'.format(len(nome))) #Funçao len() para contagem total
print('Contagem de espaços: {}'.format(nome.count(' '))) #Função count(' ') para contagem dos espaços
print('Número de caracteres (sem contar espaços): {}\n'.format(len(nome)-nome.count(' '))) #len() - count()

nomediv = nome.split() #Divisão do nome em lista[]

print('O primeiro nome é {} e possui {} letras'.format(nomediv[0], len(nomediv[0]))) #Função len() para contar caracteres do item 0 da lista

