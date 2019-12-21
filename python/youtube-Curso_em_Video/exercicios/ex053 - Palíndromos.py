# Desafio 53
# Crie um programa que leia uma frase qualquer e diga se ela é um palíndromo, desconsiderando os espaços e acentos.
# Exemplos:
# APOS A SOPA
# A SACADA DA CASA
# A TORRE DA DERROTA
# O LOBO AMA O BOLO
# ANOTARAM A DATA DA MARATONA

frase = input('Insira uma frase para verificar se é um PALÍNDROMO: ').strip().upper() #Strip elimina espaços antes e depois
s = len(frase)
b = 0
print('Quantidade de caracteres: {}'.format(len(frase)))
print('\nVocê digitou a frase: \033[1;30m{}\033[m\n'.format(frase))
print('\033[34mVisualizando na VERTICAL:\033[m')
for c in range(0, len(frase)):
    print('Frase na ordem: \033[1;30m{}\033[m - Frase invertida: \033[34m{}\033[m'.format(frase[c], frase[s-1]))
    s -= 1

# Tratando a frase digitada
palavras = frase.split() # Separa a frase em listas (palavras)
junto = ''.join(palavras) # Junta as palavras na lista concatenando sem espaço algum (junta tudo '')
inverso = '' # Variável para varrer string ao contrário. Vazio no início.
print('\n')
for letra in range(len(junto)-1, -1, -1): # Primeira letra é 0, porém temos que ir até -1 descrescendo de -1 em -1.
    inverso = inverso + junto[letra] # String inverso antes vazia recebe letra por letra ao contrário, na ordem correta...
    #inverso += junto[letra]
print('Na ordem: \033[1;30m{}\033[m - Invertida: \033[34m{}\033[m'.format(junto, inverso))
if junto == inverso:
    print('Temos um PALÍNDROMO.')
else:
    print('Não é PALÍNDROMO.')


