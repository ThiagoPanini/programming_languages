#Aula 13 - Laço de repetição for

print('\n"Lendo um valor e usando-o como critério dentro do laço for"')
n = int(input('Digite um número: '))
for c in range(0, n+1):
    print(c)
print('FIM')

print('\n"Configurando laço for através de variáveis"')
i = int(input('Início: '))
f = int(input('Fim: '))
p = int(input('Passo: '))
for c in range(i, f+1, p): # f+1 pois o número colocado nesse local não é considerado pelo Python (sempre 1 a menos)
    print(c)
print('FIM')

print('\n"Somando números dentro de um laço"')
s = 0
for c in range(0, 4):
    n = int(input('Digite um número: '))
    s = s + n # ou s += n
print('O somatório de todos os valores foi {}.'.format(s))