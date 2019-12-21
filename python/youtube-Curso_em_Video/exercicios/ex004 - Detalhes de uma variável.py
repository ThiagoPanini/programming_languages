#Desafio 04 - métodos is. algo
#Fazer um programa que leia algo pelo teclado e mostre seu tipo primitivo e todas as informações possíveis

n = input('Digite algo: ')

print('O tipo primitivo desse valor é', type(n))
print('É numérico?', n.isnumeric())
print('É alfabético?', n.isalpha())
print('Só tem espaços?', n.isspace())
print('É alfanumérico?', n.isalnum())
print('É maiúsculo?', n.isupper())
print('É minúsculo?', n.islower())
print('Capitalizado?', n.istitle())