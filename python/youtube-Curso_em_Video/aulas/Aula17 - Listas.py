# LISTAS

num = [2, 5, 9, 1]
num[2] = 3  # Listas são mutáveis!
num.append(7)   # Adicionando elementos
num.sort()  # Ordena elementos
num.sort(reverse=True)  # Imprime em ordem reversa
num.insert(2, 0)    # Adiciona o elemento 0 na posição 2
num.pop()   # Elimina (e retorna) o último valor da lista
num.pop(2)  # Faz a mesma coisa que o comando anterior, porém ao índice 2
num.remove(2)   # Remove o valor número 2 da lista (primeira aparição)
"""O comando .remove() dá erro caso o valor não se encontre na lista"""
# Boa prática:
if 4 in num:
    num.remove(4)
else:
    print('Número não se encontra na lista.')

""" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - """
valores = []    # Cria uma lista vazia
# Também poderia ser:
# valores = list()
valores.append(5)
valores.append(9)
valores.append(4)

# Função enumerate() retorna o índice e o valor em forma de tupla de dois elementos
for c, v in enumerate(valores):
    print(f'Na posição {c} encontrei o valor {v}!')
print('Cheguei ao final da lista.')

# Preenchendo lista através do input()
for cont in range(5):
    valores.append(int(input('Digite um valor: ')))

""" - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - """
a = [2, 3, 4, 7]
b = a   # Ambas apontam para o mesmo objeto lista da memória. Alterações refletem em ambas

b = a[:]    # Agora as alterações em "a" não refletem em "b" e vice-versa. [:] copia os itens