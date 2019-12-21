#include <stdio.h>

double pi(int t) {
	int i;
	int sinal=1;
	double soma = 0;
	
	for (i=1; i<=t; i++) {
		soma += 1.0/(2*i-1)*sinal;
		sinal *= -1;
	}
	return 4*soma;
}

void main() {
	printf("%lf\n", pi(1));
	printf("%lf\n", pi(10));
	printf("%lf\n", pi(100));
	printf("%lf\n", pi(1000));
}
