#Desafio 26
#Faça um programa que leia uma frase pelo teclado e mostre:

# - Quantas vezes aparece a letra "A".
# - Em que posição ela aparece a primeira vez.
# - Em que posição ela aparece a última vez.

frase = input('Digite uma frase: ').strip().upper()

print('O caractere "a" aparece {} vezes em sua frase'.format(frase.count('A')))
print('O caractere "a" aparece pela primeira vez na posição número {}'.format(frase.find('A')+1))
print('O caractere "a" aparece pela última vez na posição número {} da string'.format(frase.rfind('A')+1))

#find('A')+1 significa mostrar a posição da letra conforme lemos, e não conforme o Python entende (Python começa do 0)
#rfind('A' significa começar pela DIREITA (RIGHT) a procura pela letra 'A'