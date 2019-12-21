#Desafio 38
#Escreva um programa que leia dois números inteiros e compare-os mostrando na tabela a seguinte mensagem:
# O primeiro valor é MAIOR
# O segundo valor é MAIOR
# Não existe valor maior, os dois são iguais

cores = {'limpa':'\033[m', 'negativo':'\033[7;30m', 'branconegrito':'\033[1;30m'}
n1 = int(input('Insira o primeiro valor inteiro: '))
n2 = int(input('Insira o segundo valor inteiro: '))

if n1 > n2:
    print('Número {}{}{} é MAIOR que o número {}{}{}'.format(cores['branconegrito'], n1 , cores['limpa'],
                                                             cores['branconegrito'], n2 , cores['limpa']))
elif n2 > n1:
    print('Número {}{}{} é MAIOR que o número {}{}{}'.format(cores['branconegrito'], n2, cores['limpa'],
                                                             cores['branconegrito'], n1, cores['limpa']))
else:
    print('Os números {}{}{} e {}{}{} são {}IGUAIS{}!'.format(cores['branconegrito'], n1 , cores['limpa'],
                                                             cores['branconegrito'], n2 , cores['limpa'],
                                                              cores['negativo'], cores['limpa']))