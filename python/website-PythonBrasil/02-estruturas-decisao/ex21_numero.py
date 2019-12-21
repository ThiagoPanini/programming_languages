"""2.21 Faça um Programa que leia um número inteiro menor que 1000 e imprima
a quantidade de centenas, dezenas e unidades do mesmo.
Observando os termos no plural a colocação do "e", da vírgula entre outros.
Exemplo:   a. 326 = 3 centenas, 2 dezenas e 6 unidades   b. 12  = 1 dezena e 2 unidades
   Testar com: 326, 300, 100, 320, 310,305, 301, 101, 311, 111, 25, 20, 10, 21, 11, 1, 7 e 16 """

num = int(input('Digite um número: '))
fixo = num
print(f'O número {fixo} possui: ', end=' ')
if num >= 1000:
    milhar = num // 1000
    print(f'{milhar} milhar(es).', end=' ')
    num -= milhar * 1000
if num >= 100:
    centena = num // 100
    print(f'{centena} centena(s).', end=' ')
    num -= centena * 100
if num >= 10:
    dezena = num // 10
    print(f'{dezena} dezena(s).', end=' ')
    num -= dezena * 10
if num >= 1:
    unidade = num
    print(f'{unidade} unidade(s).')
