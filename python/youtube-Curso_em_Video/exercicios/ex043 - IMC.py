#Deasfio 43
#Desenvolva uma lógica que leia o peso e a altura de uma pessoa, calcule seu IMC e mostre seu status, conforme tabela abaixo:

# - Abaixo de 18.5 = Abaixo do peso;
# - Entre 18.5 e 25 = Peso ideal;
# - De 25 até 30 = Sobrepeso;
# - De 30 até 40 = Obesidade;
# - Acima de 40 = Obesidade mórbida.

import time

peso = float(input('Insira seu peso: '))
altura = float(input('Insira sua altura em metros: '))
imc = peso/(altura**2)
print('Seu IMC é {:.2f}'.format(imc))
print('Verificando a categoria...')
time.sleep(2)

if imc < 18.5:
    print('Você está ABAIXO DO PESO.')
elif 18.5 <= imc < 25:
    print('Você está em seu PESO IDEAL.')
elif 25 <= imc <= 30:
    print('Você está com SOBREPESO.')
elif 30 < imc <= 40:
    print('Você está com OBESIDADE.')
else:
    print('Você está com OBESIDADE MÓRBIDA.')