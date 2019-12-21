#Operadores Aritméticos

n1 = int(input('Um valor: '))
n2 = int(input('Outro valor: '))
#print('A soma vale {}'.format(n1+n2)) -> também é possível colocar operações dentro do método .format()

s = n1+n2
m = n1*n2
d = n1/n2
d1 = n1//n2
e = n1**n2

print('A soma vale {} \n o produto vale {}\n a divisão vale {:.3f}'.format(s,m,d,), end=' >>> ')
print('A divisão inteira vale {} e a potência vale {}'.format(d1,e))

#{:.3f} = 3 casas decimais ponto flutuante
#\n = quebra de linha
#, end=' ' = junção de duas linhas