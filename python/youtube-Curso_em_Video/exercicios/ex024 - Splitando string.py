#Desafio 24
#Crie um programa que leia o nome de uma a cidade e diga se ela começa ou não com o nome "SANTO"

cidade = input('Insira o nome de uma cidade: ')
splitcity = cidade.split()
print('SANTO' in splitcity[0].upper())