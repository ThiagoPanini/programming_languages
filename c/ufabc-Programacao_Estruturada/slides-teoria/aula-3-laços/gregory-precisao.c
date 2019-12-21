#include <stdio.h>

double pi(double pre) {
	int i = 1;
	int sinal=1;
	double soma = 0.0;
	
	while((1.0/(2*i-1)) > pre) {
		soma += 1.0/(2*i-1)*sinal;
		sinal *= -1;
		printf("%.10lf\n", soma);
		i++;
	}
	
	return 4*soma;
}

void main() {
	printf("%.10lf\n", pi(0.1));
	printf("%.10lf\n", pi(0.00001));
}
