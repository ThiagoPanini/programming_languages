# Desafio 64
# Crie um programa que leia vários números inteiros pelo teclado.
# O programa só vai parar quando o usuário digitar o valor 999, que é a condição de parada.
# No final, mostre quantos números foram digitados e qual foi a soma entre eles (desconsiderando o flag).

n = cont = soma = 0
while n != 999:
    n = int(input('Digite um número [999 para SAIR]: '))
    if n != 999:
        soma += n
        cont += 1
print('\nForam digitados {} números, sendo a soma entre eles igual a {}.'.format(cont, soma))
print('\nFIM')

# OUTRA FORMA DE RESOLVER...

"""n = int(input('Digite um número [999 para SAIR]: '))  # Lê o número primeiro
while n != 999:
    soma += n
    cont += 1
    n = int(input('Digite um número [999 para SAIR]: ')) # Pergunta novamente no final do laço.
    # Dessa forma o 999 não é considerado na contagem e na soma. Não é necessário fazer cont-1 ou soma-999
print('\nForam digitados {} números, sendo a soma entre eles igual a {}.'.format(cont, soma))
print('\nFIM')"""

