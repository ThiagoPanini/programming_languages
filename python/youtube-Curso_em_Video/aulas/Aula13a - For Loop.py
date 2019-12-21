#Aula13 - Laços de repetição - for

print('\n"Printando "Oi" 5 vezes na tela: range(1, 6)"')
for c in range(1, 6):
    print('Oi')
print('FIM')
# O comando acima escreve 5 VEZES a palavra 'Oi' na tela, pois no formato range(1, 6), o segundo número não é considerado.

print('\n"Printando "Oi" 6 vezes na tela: range(0, 6)"')
for c in range(0, 6):
    print('Oi')
print('FIM')
# O comando acima escreve 6 VEZES a palavra 'Oi' pois agora o range é de ZERO até SEIS (seis vezes).

print('\n"Contando a variável c de 1 até 6: range(1, 7)"')
for c in range(1, 7): # Contagem de 1 até 6 (apesar de estar 7 no segundo argumento do range)
    print(c)
print('FIM')

print('\n"Contando decrescente de 6 até 1: range(6, 0, -1)"')
for c in range(6, 0, -1): # Terceiro argumento do range = especifica o critério de contagem (neste caso, decrescente)
    print(c)
print('FIM')

print('\n"Contando crecente de 0 até 6 indo de 2 em 2: range(0, 7, 2)"')
for c in range(0, 7, 2): # Terceiro argumento do range = especifica o critério de contagem (neste caso, de 2 em 2)
    print(c)
print('FIM')