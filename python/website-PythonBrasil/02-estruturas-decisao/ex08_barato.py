"""2.8 Faça um programa que pergunte o preço de três  produtos e informe qual produto você deve comprar,
sabendo que a decisão é sempre pelo mais barato. """

produto1 = float(input('Preço do produto 1: R$'))
produto2 = float(input('Preço do produto 2: R$'))
produto3 = float(input('Preço do produto 3: R$'))

if produto1 < produto2 < produto3 or produto1 < produto3 < produto2:
    menor = produto1
elif produto2 < produto1 < produto3 or produto2 < produto3 < produto1:
    menor = produto2
elif produto3 < produto1 < produto2 or produto3 < produto2 < produto1:
    menor = produto3

print(f'\nO produto mais barato custa R${menor:.2f}.')