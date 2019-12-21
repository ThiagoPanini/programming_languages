"""3.36 Desenvolva um programa que faça a tabuada de um número qualquer inteiro que será digitado pelo usuário, mas a
tabuada não deve necessariamente iniciar em 1 e terminar em 10, o valor inicial e final devem ser informados também pelo
 usuário. Obs: Você deve verificar se o usuário não digitou o final menor que o inicial. """

num = int(input('Digite um número para ver sua TABUADA: '))
n1 = int(input('Intervalo 1: '))
n2 = int(input('Intervalo 2: '))
while n2 < n1:
    n2 = int(input(f'Intervalo 2 deve ser maior que o Intervalo 1 ({n1}) Digite novamente: '))

for c in range(n1, n2+1):
    print(f'{num} x {c} = {num*c:2}')