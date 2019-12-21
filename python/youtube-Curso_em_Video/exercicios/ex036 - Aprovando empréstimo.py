#Desafio 36 - Escreva um programa para aprovar o empréstimo bancário para a compra de uma casa.
#O programa vai perguntar o valor da casa, o salário do comprador e em quantos anos ele vai pagar.
#Calcule o valor da prestação mensal, sabendo que ela não pode exceder 30% do salário ou então o empréstimo será negado.

casa = float(input('Insira o valor da casa a ser ser financiada: R$'))
salario = float(input('Insira o salário do titular do empréstimo: R$'))
anos = int(input('Em quantos anos pretende quitar o financiamento? '))

mensal = casa/(anos*12)

if mensal > salario*0.30:
    print('Infelizmente o empréstimo foi \033[1;31mnegado\033[m pois a parcela de R${:.2f} excede 30% de seu salário de R${:.2f}'.format(mensal, salario))
else:
    print('Empréstimo \033[1;32mconcedido\033[m! O valor da parcela será de R${:.2f} e será quitada em {} anos'.format(mensal, anos))
    print('O valor da parcela equivale a {:.2f}% de seu salário mensal.'.format(mensal*100/salario))

#, end='' faz com que os prints se juntem em uma única linha