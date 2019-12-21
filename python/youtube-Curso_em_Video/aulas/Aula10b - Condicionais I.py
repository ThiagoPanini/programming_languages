#Condições IF e ELSE - Média

n1 = float(input('Primeira nota: '))
n2 = float(input('Segunda nota: '))
m = (n1+n2)/2

print('Sua média foi {:.1f}'.format(m))

if m >= 6:
    print('Sua média foi boa, PARABÉNS!')
else:
    print('Sua média foi ruim, ESTUDE MAIS!')

#Utilizando if simplificado

print('\nPARABÉNS' if m>=6 else 'ESTUDE MAIS!')
