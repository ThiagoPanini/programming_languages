"""Retomada Curso em Vídeo - Aula 16
- - - TUPLAS (VARIÁVEIS COMPOSTAS)- - -"""

# Inicialização de tuplas
lanche = ("Hambúrguer", "Suco", "Pizza", "Pudim")

# Também é possível criar a tupla sem parênteses
lanches = 'Hambúrguer', 'Suco', 'Pizza', 'Pudim'

# Indexação em python sempre inicia em 0
print(lanche[0])  # = 'Hambúrguer'
print(lanche[-1])  # = 'Pudim'
print(lanche[-2])  # = 'Pizza'

# Fatiamento também funciona (primeiro índice ÍNCLUSIVO e segundo índice EXCLUSIVO)
print(lanche[1:3])  # = 'Suco', 'Pizza'
print(lanche[2:])  # = 'Pizza', 'Pudim'

""" - - - AS TUPLAS SÃO IMUTÁVEIS - - - """
# Não é possível alterar valores dentro de tuplas

# Passando por elementos
# Iteração com operador in
for comida in lanche:
    print(f'Eu vou comer {comida}!')

# Iteração com operador in
for i in range(len(lanche)):
    print(f'Eu vou comer {lanche[i]}')

# Utilizando a função enumerate()
for i, comida in enumerate(lanche):
    print(f'Eu vou comer {comida} na posição {i}!')

# Colocando elementos em ordem
print(sorted(lanche))

# Outros métodos

"""
.count(elemento) = retorna quantas vezes o elemento elemento se repete
.index(elemento, [a partir de (default = 0)]) = retorna a primeira posição de aparição do elemento elemento
.del(elemento) = deleta a primeira aparição do elemento elemento
"""