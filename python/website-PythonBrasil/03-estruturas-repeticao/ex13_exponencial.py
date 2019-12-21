"""3.13 Faça um programa que peça dois números, base e expoente,
calcule e mostre o primeiro número elevado ao segundo número. Não utilize a função de potência da linguagem."""

r = 'S'
while r in 'Ss':
    base = int(input('Digite a base: '))
    exp = int(input('Digite o expoente: '))
    print(f'{base} elevado a {exp} é igual a {base**exp}')
    r = input('Deseja continuar? [S/N] ').strip().upper()[0]
print('Fim.')

