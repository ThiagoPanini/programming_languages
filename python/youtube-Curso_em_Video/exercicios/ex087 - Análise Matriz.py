"""Exercício 87
Aprimore o desafio anterior, mostrando no final:
(A) A soma de todos os valores pares digitados.
(B) A soma dos valores da terceira coluna.
(C) O maior valor da segunda linha.
"""

matriz = [[], [], []]
soma_pares = maior_segunda_linha = soma_terceira_coluna = 0

for i in range(3):
    for j in range(3):
        matriz[i].append(input(f'Digite um valor para o elemento [{i+1}][{j+1}]: '))

# Imprimindo matriz
print('-'*10)
for i in range(3):
    for j in range(3):
        print(f'[{matriz[i][j]:^5}]', end='')
        if matriz[i][j] % 2 == 0:
            soma_pares += matriz[i][j]
    print()
print('-'*10)

# Analisando a Matriz
print(f'A soma dos pares é igual é {soma_pares}')

# Analisando coluna 3
for l in range(3):
    soma_terceira_coluna += matriz[l][2]
print(f'A soma dos elementos da terceira coluna é igual a {soma_terceira_coluna}')

# Analisando segunda linha
for c in range(3):
    if c == 0:
        maior_segunda_linha = matriz[1][c]
    elif matriz[1][c] > maior_segunda_linha:
        maior_segunda_linha = matriz[1][c]
print(f'O maior valor da segunda linha é igual a {maior_segunda_linha}')
