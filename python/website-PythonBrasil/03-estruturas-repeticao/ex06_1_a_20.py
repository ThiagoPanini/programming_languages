"""3.6 Faça um programa que imprima na tela os números de 1 a 20, um abaixo do outro.
Depois modifique o programa para que ele mostre os números um ao lado do outro.  """

print('-'*18)
print('Números de 1 a 20')
print('-'*18)

for c in range(1, 21):
    print(c)
print('Outra forma:')
for c in range(1, 21):
    print(c, end=' ')