"""3.30 O Sr. Manoel Joaquim acaba de adquirir uma panificadora e pretende implantar a metodologia da tabelinha,
que já é um sucesso na sua loja de 1,99. Você foi contratado para desenvolver o programa que monta a tabela de preços
de pães,  de 1 até 50 pães,  a partir do preço do pão informado pelo usuário"""

pao = float(input('Preço do pão: R$'))
print(f'Tabela de preços para o pão a R${pao:.2f}')
for c in range(1, 51):
    print(f'{c:<2} - R${c*pao:5.2f}')