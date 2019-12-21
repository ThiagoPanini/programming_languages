#include <stdio.h>

int main() {
	
	int i, j;
	int col = 11;
	int aux;
	int cont = 1;
	double M[12][12], soma=0.0;
	char operacao;
	
	scanf("%c", &operacao);
	for(i=0;i<12;i++){
		for(j=0;j<12;j++){
			scanf("%lf", &M[i][j]);
			if ((j>i) && (i+j>11))
				soma += M[i][j];
		}
	}
	
	if (operacao == 'S') 
		printf("%.1lf\n", soma);
	else if (operacao == 'M') 
		printf("%.1lf\n", soma/30.0);
	
	return 0;
}
