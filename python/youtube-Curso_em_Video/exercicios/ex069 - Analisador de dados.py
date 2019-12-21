"""Desafio 69
Crie um programa que leia a idade e o sexo de várias pessoas.
A cada pessoa cadasttrada, o programa deverá perguntar se o usuário quer ou não continuar. Ao final, mostre:
a) Quantas pessoas tem mais de 18 anos;
b) Quantos homens foram cadastrados;
c) Quantas mulheres tem menos de 20 anos."""

idade = maior18 = mulher20 = homens = 0
sexo = ''
r = 'S'
i = 1
print('-'*20)
print('PAINEL DE CADASTRO')
print('-'*20)
while r in 'Ss':
    print('='*20)
    print(f'CADASTRO DA PESSOA {i}')
    print('='*20)
    idade = int(input('Idade da pessoa {}: '.format(i)))
    sexo = str(input('Sexo da pessoa {}: [M/F] '.format(i))).strip().upper()[0] # Pega apenas a primeira letra
    while sexo not in 'MmFf':
        sexo = str(input('Valor inválido, digite M ou F: '))
    if idade >= 18:
        maior18 += 1
    if sexo in 'Mm':
        homens += 1
    if sexo in 'Ff' and idade < 20:
        mulher20 += 1
    r = input('Deseja continuar? [S/N] ').strip().upper()
    while r not in "SsNn":
        r = input('Deseja continuar? [S/N] ').strip().upper()
    if r not in 'Ss':
        break
    i += 1
print('-'*11)
print('RESULTADOS')
print('-'*11)
print(f'Foram cadastradas {i} pessoas no total.')
print(f'Destas, {homens} Homem(ns) e {i-homens} Mulher(es).')
print(f'Temos {maior18} pessoa(s) maior(es) de 18 anos.')
print(f'Há {mulher20} mulher(es) com menos de 20 anos.')
