"""3.15 A série de Fibonacci é formada pela seqüência 1,1,2,3,5,8,13,21,34,55,...
Faça um programa capaz de gerar a série até o n−ésimo termo."""

# Programa já realizado no curso do Gustavo Guanabara.

print('-'*10)
print('FIBONACCI')
print('-'*10)

termo = int(input('Até qual termo deseja visualizar a série? '))

anterior = 1
atual = 1
proximo = 2

for c in range(termo):
    if c == termo-1:
        print(anterior, end='.')
    else:
        print(anterior, end=' -> ')
        anterior = atual
        atual = proximo
        proximo = anterior + atual
