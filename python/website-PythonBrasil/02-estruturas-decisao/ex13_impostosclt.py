vlrhora = float(input('Valor por hora: R$'))
horames = int(input('Horas trabalhadas por mês: '))

sbruto = vlrhora*horames

inss = 15/100
fgts = 11/100
sindicato = 3/100
print(f'Salário bruto: R${sbruto:.2f}')

if sbruto <= 900:
    ir = 1
elif 900 < sbruto <= 1500:
    ir = 5/100
elif 1500 < sbruto <= 2500:
    ir = 10/100
elif sbruto > 2500:
    ir = 20/100

sliquido = sbruto - (sbruto*ir) - (sbruto*inss) - (sbruto*fgts) - (sbruto*sindicato)

print(f'IR ({ir*100}%): R${sbruto*ir:.2f}')
print(f'INSS: ({inss*100}%): R${sbruto*inss:.2f}')
print(f'FGTS: ({fgts*100}%): R$ {sbruto*fgts:.2f}')
print(f'SINDICATO: ({sindicato*100}%): R${sbruto*sindicato:.2f}')

print(f'\nSalário líquido: R${sliquido:.2f}')

