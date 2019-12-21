# Desafio 55
# Faça um programa que leia o peso de 5 pessoas. No final, mostre qual foi o maior e o menor peso lidos.
maior = 0
#menor = 1000 # Gambiarra que Guanabara ensina a evitar
menor = 0 # Ambos começam com 0
for c in range(0, 5):
    peso = float(input('Insira o peso em Kg da pessoa {}: '.format(c + 1)))
    if c == 0: # Se é a primeira verificação
        maior = peso
        menor = peso # Ambos são os maiores e menores pois é o primeiro valor digitado
    else: # Se não é a primeira verificação, aí eu vejo qual é o maior e qual é o menor
        if peso > maior:
            maior = peso
        if peso < menor:
            menor = peso

print('\n')
print('Maior peso: {:.2f}Kg'.format(maior))
print('Menor peso: {:.2f}Kg'.format(menor))