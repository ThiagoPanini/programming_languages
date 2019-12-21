"""3.4 Supondo que a população de um país A seja da ordem de 80000 habitantes com uma taxa anual de crescimento de 3%
e que a população de B seja 200000 habitantes com uma taxa de crescimento de 1.5%. Faça um programa que calcule e
escreva o número de anos necessários para que a população do país A ultrapasse ou iguale a população do país B,
mantidas as taxas de crescimento. """

import datetime

popA = 80000
txA = 3/100
popB = 200000
txB = 1.5/100
ano = 0

print('-'*9)
print('POPULAÇÃO')
print('-'*9)

while popA <= popB:
    popA += popA*txA
    popB += popB*txB
    ano += 1
print(f'Foram necessários {ano} ano(s) para a População A alcançar ou ultrapassar a População B.')
print(f'Após este(s) {ano} ano(s), temos:\n'
      f'População A: {popA:.0f} habitantes.\n'
      f'População B: {popB:.0f} habitantes.')