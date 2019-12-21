#Desafio 12 - Faça um algoritmo que leia o preço do produto e mostre seu novo preço com 5% de desconto

v = float(input('Digite o preço do produto: '))
d = float(input('Digite a porcentagem a ser aplicada como desconto: '))
desc = v*(d/100)
print('Desconto no valor de R$ {:.2f}'.format(desc))
print('O valor com o desconto de {}% é igual a R$ {:.2f}'.format(d, v-desc))

