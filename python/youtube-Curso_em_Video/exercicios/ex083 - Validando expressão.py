"""Exercício 83
Crie um programa onde o usuário digite uma expressão qualquer que use parênteses.
Seu aplicativo deverá analisar se a expressão passada está com parênteses abertos e fechados na ordem correta.
"""

print('-'*25)
print('{:^25}'.format('VALIDANDO EXPRESSÃO MATEMÁTICA'))
print('-'*25)

expressao = input('Digite a expressão: ')
if expressao.count('(') == expressao.count(')'):
    print('Expressão válida.')
else:
    print('Expressão inválida.')


# Resolução Guanabara

expr = input('Digite a expressão: ')
pilha = []
for simb in expr:
    if simb == '(':
        pilha.append('(')
    elif simb == ')':
        if len(pilha) > 0:
            pilha.pop()
        else:
            pilha.append(')')
            break
if len(pilha) == 0:
    print('Sua expressão está correta.')
else:
    print('Sua expressão é inválida.')