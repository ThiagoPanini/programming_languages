#Desafio 11 - Faça um programa que leia a largura e altura de uma parede em metros,
#calcule sua área e a quantidade de tinta necessária para pinta-la, sabendo que cada L de tinta pinta 2m²

l = float(input('Digite a largura da parede, em metros: '))
h = float(input('Digite a altura da parede, em metros: '))
a = l*h
print('A área da parede é igual a {:.2f} m²'.format(a))
print('\nInformação: cada litro de tinta pinta uma área de 2m²')
print('\nLogo, para pintar uma área de {} m², será(ão) necessário(s) {:.2f} litro(s) de tinta'.format(a, (a/2)))

