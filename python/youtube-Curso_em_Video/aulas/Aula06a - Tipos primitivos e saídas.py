n1 = int(input('Digite um valor: '))
print(type(n1))

n2 = int(input('Digite um valor: '))
print(type(n2))

s = n1+n2

#print('A soma entre',n1,'e',n2,'vale',s)

#Temos uma forma melhor de apresentar a soma... novo modo do Python3

print('A soma entre {0} e {1} vale {2}'.format(n1,n2,s))
#Podemos também numerar as máscaras para ajudar na seleção das variáveis no método .format()