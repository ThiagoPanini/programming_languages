"""2.30   Se o cliente comprar mais de 8 Kg em frutas ou o valor total da compra ultrapassar R$ 25,00,
receberá ainda um desconto de 10% sobre este total. Escreva um algoritmo para ler a quantidade (em Kg) de morangos
e a quantidade (em Kg) de maças adquiridas e escreva o valor a ser pago pelo cliente.

 Uma fruteira está vendendo frutas com a seguinte tabela de preços:
    Até 5 Kg               Acima de 5 Kg
 Morango R$ 2,50 por Kg       R$ 2,20 por Kg
 Maçã R$ 1,80 por Kg          R$ 1,50 por Kg"""

kgmor = float(input('Quantos Kg de Morango? '))
kgmaca = float(input('Quantos Kg de Maçã? '))
somakg = kgmor + kgmaca

vlrmor = kgmor*2.25 if kgmor <= 5 else kgmor*2.20
vlrmaca = kgmaca*1.80 if kgmaca <= 5 else kgmaca*1.50
somavlr = vlrmor + vlrmaca

print(f'Valor do Morango: R${vlrmor:.2f}')
print(f'Valor da Maçã: R${vlrmaca:.2f}')
print(f'Total em Reais: R${somavlr:.2f}')
print(f'Total em Kg: {somakg:.2f}Kg')

if somakg > 8 or somavlr > 22.50:
    print(f'Valor total com DESCONTO de 10%: R${somavlr*0.90:.2f}')