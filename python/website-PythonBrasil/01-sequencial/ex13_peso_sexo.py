"""1.13 Tendo como dados de entrada a altura e o sexo de uma pessoa, construa um
algoritmo que calcule seu peso ideal, utilizando as seguintes fórmulas:
 a. Para homens: (72.7*h) - 58
 b. Para mulheres: (62.1*h) - 44.7 (h = altura)
 c. Peça o peso da pessoa e informe se ela está dentro, acima ou
abaixo do peso."""

h = float(input('Insira sua altura em metros: '))
genero = input('Digite o sexo: [M/F] ').strip().upper()[0]

if genero in 'Mm':
    pesoideal = (72.7*h) - 58
elif genero in 'Ff':
    pesoideal = (62.1*h) - 44.7

peso = float(input('Digite seu peso em kg: '))

if peso > pesoideal:
    print(f'Seu peso ideal é de {pesoideal:.2f}kg e você está ACIMA do peso.')
else:
    print(f'Seu peso ideal é de {pesoideal:.2f}kg e você está DENTRO do peso.')