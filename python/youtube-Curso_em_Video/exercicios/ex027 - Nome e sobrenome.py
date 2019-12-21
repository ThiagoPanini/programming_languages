#Desafio 27
#Faça um programa que leia o nome completo de uma pessoa,
#mostrando em seguida o primeiro e o último nome separadamente.

#Ex: Ana Maria de Souza
#primeiro = Ana
#último = Souza

nome = input('Digite o nome completo: ').title().strip().split()

#print(len(nome)) #Como eu splitei o nome, o len agora conta a quantidade de itens na minha lista

print('Primeiro nome: {}'.format(nome[0]))
print('Último nome: {}'.format(nome[len(nome)-1]))