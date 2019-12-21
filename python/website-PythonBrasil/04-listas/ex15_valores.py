"""4.15 Faça um programa que leia um número indeterminado de valores, correspondentes a notas, encerrando a entrada de
dados quando for informado um valor igual a -1 (que não deve ser armazenado). Após esta entrada de dados, faça:
   a. Mostre a quantidade de valores que foram lidos;
   b. Exiba todos os valores na ordem em que foram informados, um ao lado do outro;
   c. Exiba todos os valores na ordem inversa à que foram informados, um abaixo do outro;
   d. Calcule e mostre a soma dos valores;
   e. Calcule e mostre a média dos valores;
   f. Calcule e mostre a quantidade de valores acima da média calculada;
   g. Calcule e mostre a quantidade de valores abaixo de sete;
   h. Encerre o programa com uma mensagem"""


class CorPy:
    azul = '\033[1;34m'
    end = '\033[m'


def cabecalho(texto):
    """Imprime um cabeçalho/título de acordo com o texto inserido Objetivo: facilitar a visualização do programa"""
    print('-'*40)
    print('{:^49}'.format(CorPy.azul + texto + CorPy.end))
    print('-'*40)


def preencheLista():
    lista = []
    while True:
        valor = float(input('Insira um valor para a lista: '))
        if valor == -1:
            break
        lista.append(valor)
    return lista


def quantidade(lista):
    """Imprime no prompt a quantidade de valores inseridos na lista"""
    print(f'Quantidade de valores lidos: {len(lista)}')


def ordemEntrada(lista):
    """Imprime no prompt os valores na ordem em que foram inseridos"""
    i = 1
    soma = 0
    for x in lista:
        print(f'{i}º: {x}')
        i += 1
        soma += x
    media = soma/len(lista)
    return soma, media


def ordemInversa (lista):
    i = len(lista)
    for x in range(i):
        print(f'{i}º: {lista[i-1]}')
        i -= 1


def acimaMediaMenor7(lista, media):
    cont = menorsete = 0
    print('Valores acima da média: ', end='')
    for x in lista:
        if x > media:
            cont += 1
            print(x, end=', ')
        if x < 7:
            menorsete += 1
    print(f'\nTotal: {cont}')
    return menorsete


menu = {'a': 'Quantidade de valores lidos',
        'b': 'Valores na ordem de entrada',
        'c': 'Valores na ordem inversa',
        'd': 'Soma dos valores',
        'e': 'Média dos valores',
        'f': 'Valores acima da média',
        'g': 'Quantidade de valores menores que sete',}

bd = preencheLista()
cabecalho(menu['a'])
quantidade(bd)
cabecalho(menu['b'])
soma, media = ordemEntrada(bd)
cabecalho(menu['c'])
ordemInversa(bd)
cabecalho(menu['d'])
print(f'Soma: {soma}')
cabecalho(menu['e'])
print(f'Média: {media:.2f}')
menorsete = acimaMediaMenor7(bd, media)
cabecalho(menu['f'])
print(f'Quantidade de valores menores que sete: {menorsete}')
cabecalho('FIM')

# join + split = chamar tudo num FOR único
# Nome das funções = mesmo nome da chave do dicionario menu