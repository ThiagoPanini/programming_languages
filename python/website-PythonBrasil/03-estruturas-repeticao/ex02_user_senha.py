"""3.2 Faça um programa que leia um nome de usuário e a sua senha e não aceite a senha igual ao nome do usuário,
mostrando uma mensagem de erro e voltando a pedir as informações"""

print('-'*15)
print('CADASTRO USUÁRIO')
print('-'*15)

user = input('Digite um nome de usuário: ')
senha = input('Digite uma senha: ')

while user == senha:
    print('Erro. A senha não pode ser igual ao nome de usuário.')
    change = int(input('Deseja trocar: \n[ 1 ] Usuário\n[ 2 ] Senha\n'))
    while change not in range(1,3):
        change = int(input('Opção inválida. Selecione para trocar: \n[ 1 ] Usuário\n[ 2 ] Senha'))
    if change == 1:
        user = input('Digite um nome de usuário: ')
    else:
        senha = input('Digite uma senha: ')
print('Usuário cadastrado com sucesso!\n'
      f'Login: {user}\n'
      f'Senha: {senha}')