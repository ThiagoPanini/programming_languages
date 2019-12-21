"""1.14 João Papo-de-Pescador, homem de bem, comprou um microcomputador para
controlar o rendimento diário de seu trabalho. Toda vez que ele traz um peso de
peixes maior que o estabelecido pelo regulamento de pesca do estado de São
Paulo (50 quilos) deve pagar uma multa de R$ 4,00 por quilo excedente. João
precisa que você faça um programa que leia a variável peso (peso de peixes) e
verifique se há excesso. Se houver, gravar na variável excesso e na variável multa
o valor da multa que João deverá pagar. Caso contrário mostrar tais variáveis com
o conteúdo ZERO."""

peso = float(input('Peso dos peixes: '))
max = 50
if peso > 50:
    excesso = peso - max
    multa = 4*excesso
    print(f'João ultrapassou {excesso}kg de peixes e terá que pagar uma multa no valor de R${multa:.2f}.')
else:
    print('O peso está dentro do limite. Não será necessário pagar multa.')