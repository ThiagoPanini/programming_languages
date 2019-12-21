# Desafio 65
# Crie um programa que leia vários números inteiros pelo teclado.
# No final da execução, mostre a média entre todos os valores e qual foi o maior e o menor valor lido.
# O programa deve perguntar ao usuário se ele quer ou não continuar a digitar valores.

n = soma = media = maior = menor = cont = 0
i = 1
r = 'S'

while r == 'S':  # Enquanto a resposta é S ou s (Sim)... # Poderia ser while r in 'Ss':
    n = int(input('Digite um número inteiro: '))    # Começa lendo um número
    if maior == menor == 0:  # Se é a primeira verificação...
        maior = menor = n  # Ambos maior e menor recebem o número. Assim evita gambiara de numero alto no menor (inicio)
    else:   # Se não é a primeira verificação (Laço acima só entrará a primeira vez. Depois cairá sempre no else).
        if n > maior:   # Verifica se o número é maior que o primeiro (ou o anterior) já inserido
            maior = n   # Verifica se o número é maior que o primeiro (ou o anterior) já inserido
        if n < menor:
            menor = n
    r = input('Deseja continuar? [S/N]: ').upper().strip()[0]
    # upper().strip()[0] = considera minúsculas, elimina espaços e considera apenas a 1ª letra da palava digitada.
    cont += 1
    soma += n

print('-'*10)
print('RESULTADOS')
print('-'*10)
print('Foram digitados {} valores'.format(cont))
print('A soma entre eles foi de {}'.format(soma))
print('A média entre eles foi de {:.2f}'.format(soma/cont))
print('O maior foi {}'.format(maior))
print('O menor foi {}'.format(menor))
print('\nFIM')
