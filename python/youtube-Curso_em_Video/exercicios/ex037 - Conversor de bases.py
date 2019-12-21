#Desafio 37
#Escreva um programa que leia um número inteiro qualquer e peça pra o usuário escolher a base de conversão:
# 1 para binário;
# 2 para octal;
# 3 para hexadecimal

num = int(input('Insira um número inteiro: '))
base = int(input('Escolha a base de conversão, sendo:\n'
                 '[ 1 ] para binário\n'
                 '[ 2 ] para octal\n'
                 '[ 3 ] para hexadecimal\n'))

if base != 1 and base != 2 and base != 3:
    base = int(input('Você inseriu um número inválido. Tente novamente: '))

if base == 1:
    print('Base binária selecionada.')
    print('{} convertido para BINÁRIO é igual a {}'.format(num, bin(num)[2:])) #[:2] = corta as duas primeiras letras do número
elif base == 2:
    print('Base octal selecionada.')
    print('{} convertido para OCTAL é igual a {}'.format(num, oct(num)[2:]))
elif base == 3:
    print('Base hexadecimal selecionada.')
    print('{} convertido para HEXADECIMAL é igual a {}'.format(num, hex(num)[2:]))
else:
    print('O número digitado continua inválido. Encerrando programa...')

#print('\n\033[7;30mAinda não há conhecimentos necessários para realizar a conversão\033[m')