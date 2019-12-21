#Desafio 23
#Faça um programa que leia um número de 0 a 9999 e mostre na tela cada um dos dígitos separados
#Ex: 'Digite um número: 1834'
# - unidade: 4
# - dezena: 3
# - centena: 8
# - milhar: 1

num = input('Digite um número de 0 a 9999: ')
qtd = (len(num)) #Quantidade de caracteres da 'string' digitada

"""print('Unidade: {}'.format(num[qtd-1])) #Printa o último caractere da string (lembrando que começa do 0)
print('Dezena: {}'.format(num[qtd-2]))
print('Centena: {}'.format(num[qtd-3])) #Pode dar erro em números menores
print('Milhar: {}'.format(num[qtd-4])) #Pode dar erro em números menores"""

#Resolvendo matematicamente: Resolução Guanabara
num = int(num)
u = num // 1 % 10 #Divisão inteira pela unidade e obter o resto da divisão = unidade
d = num // 10 % 10 #Divisão inteira pela dezena e obter o resto da divisão = dezena
c = num // 100 % 10 #Divisão inteira pela centena e obter o resto da divisão = centena
m = num // 1000 % 10 #Divisão inteira pelo milhar e obter o resto da divisão = milhar

print('Unidade: {}'.format(u))
print('Dezena: {}'.format(d))
print('Centena: {}'.format(c))
print('Milhar: {}'.format(m))
