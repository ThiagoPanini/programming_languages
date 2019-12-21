# DESFAIO 66
# Crie um programa que leia vários números inteiros pelo teclado.
# O programa só vai parar quando o usuário digitar o valor 999 que é a condição de parada.
# No final, mostre quantos números foram digitados e qual foi a soma entre els (desconsiderando o flag).

cont = soma = 0
while True:
    n = int(input('Digite um número: (999 para sair) '))
    if n == 999:
        break
    cont += 1
    soma += n
print(f'Foram digitados {cont} números e a soma entre eles vale {soma}.')