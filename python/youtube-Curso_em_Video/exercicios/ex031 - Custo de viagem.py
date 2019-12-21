#Desafio 31
#Desenvolva um programa que pergunte a distância de uma viagem em Km;
#Calcule o preço da passagem, cobrando R$0,50 por Km para viagens de até 200Km e R$0,45 para viagens mais longas.

dist = float(input('Esta viagem possui quantos Km de distância? '))
print('Viagem de {}Km\n'.format(dist))

"""if dist <= 200:
    valor = (dist*0.50)
    print('Viagem até 200Km de distância')
else:
    valor = (dist*0.45)
    print('Viagem maior que 200Km de distância')"""
#Fazendo com if simplificado;
valor = dist*0.50 if dist<=200 else dist*0.45
print('Valor a ser cobrado: R${:.2f}'.format(valor))