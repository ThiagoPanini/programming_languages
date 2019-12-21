# Desafio 56
# Desenvolva um programa que leia nome, idade e sexo de 4 pessoas. No final do programa, mostre:
# A média de idade do grupo,
# Qual é o nome do homem mais velho;
# Quantas mulheres tem menos de 20 anos.

#Luz: será que é por lista?
"""nomes = ['Nome1', 'Nome2', 'Nome3', 'Nome4']
idades = ['Idade1', 'Idade2', 'Idade3', 'Idade4']
sexo = ['Sexo1', 'Sexo2', 'Sexo3', 'Sexo4']"""

nomes = ['', '', '', '']
idades = ['', '', '', '']
sexo = ['', '', '', '']

idadevelho = 0
nomevelho = ''
mcont = 0
soma = 0
for c in range(0, 4):
    print('=-'*7)
    print('Cadastro {}:'.format(c+1))
    print('=-' * 7)
    nomes[c] = input('Nome: ')
    idades[c] = int(input('Idade: '))
    sexo[c] = input('Sexo - [ M ] para Masculino e [ F ] para Feminino: ')
    soma = soma + idades[c]
    if sexo[c] == 'M' and idades[c] > idadevelho:
        idadevelho = idades[c]
        nomevelho = nomes[c]
    if sexo[c] == 'F' and idades[c] < 20:
        mcont += 1
print('\n')
print('=-'*7)
print('Banco de Dados')
print('=-'*7)
print('Nomes: ', nomes)
print('Idades: ', idades)
print('Gêneros: ', sexo)
print('\n')
print('Somatória de idades: {}'.format(soma))
print('Média de idades: {:.0f}'.format(soma/4))
print('Homem mais velho: {}, com {} anos.'.format(nomevelho, idadevelho))
print('Há {} mulheres com menos de 20 anos.'.format(mcont))