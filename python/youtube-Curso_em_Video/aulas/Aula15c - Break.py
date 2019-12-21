#  Outras aplicações de fstrings = Interpolação de strings

nome = 'José'
idade = 33
print(f'O {nome} tem {idade} anos.')  # Escrevendo com f strings - PYTHON 3.6+
print('O {} tem {} anos.'.format(nome, idade))  # Escrevendo com o format - PYTHON 3
print('O %s tem %d anos.' % (nome, idade)) # Modo mais antigo no PYTHON 2

# Mais possibilidades...
print('-'*20)
salario = 987.3
print(f'O {nome:-^20} tem {idade} e ganha R${salario:.2f}')  # Todas as formatações são possíveis. Centralizado.
print(f'O {nome:->20} tem {idade} e ganha R${salario:.2f}')  # Alinhado a direita e complementado com traços.
print(f'O {nome:-<20} tem {idade} e ganha R${salario:.2f}')  # Alinhado a esquerda e complementado com traços.