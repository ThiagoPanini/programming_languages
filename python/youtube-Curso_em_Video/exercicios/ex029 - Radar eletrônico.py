#Deafio 29
#Escreva um programa que leia a velocidade de um caro.
#Se ele ultrapassar 80Km/h, mostre uma mensagem dizendo que ele foi multado.
#A multa vai custar R$7,00 por cada Km acima do limite

v = int(input('Insira a velocidade do carro em Km/h: '))

print('\nSua velocidade foi de {}Km/h'.format(v))
if v > 80:
    multa = (v-80)*7
    print('Você foi multado! O valor da multa é de R${:.2f}'.format(multa))
else:
    print('Não há registro de multas.')

