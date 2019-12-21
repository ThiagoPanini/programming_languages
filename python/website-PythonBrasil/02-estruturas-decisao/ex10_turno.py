"""2.10 Faça um Programa que pergunte em que turno você estuda.
Peça para digitar M-matutino ou V-Vespertino ou N- Noturno.
Imprima a mensagem "Bom Dia!", "Boa Tarde!" ou "Boa Noite!" ou "Valor Inválido!", conforme o caso. """

turno = input('Em que turno estuda? '
              '\n[M] para MATUTINO'
              '\n[V] para VESPERTINO'
              '\n[N] para NOTURNO\n').strip().upper()

if turno in 'M':
    print('Bom dia!')
elif turno in 'V':
    print('Boa tarde!')
elif turno in 'N':
    print('Boa noite!')