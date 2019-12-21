"""4.4 Faça um Programa que leia um vetor de 10 caracteres, e diga quantas consoantes foram lidas.
Imprima as consoantes."""

vetor = input('Insira uma frase de aproximadamente 10 caracteres: ')
splitado = vetor.split()
junto = ''.join(splitado)

lista = []
cont = 0
for char in junto:
    if char not in ['a', 'e', 'i', 'o', 'u']:
        lista.append(char)
        cont += 1
print(f'Qtd de consoantes: {cont}\nLista de consoantes: {lista}')
