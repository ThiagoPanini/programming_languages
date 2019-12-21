#include <stdio.h>

int x = 2;

void pares_quadrados(int n) {
	if (x<=n) {
		printf("%d^2 = %d\n", x, x*x);
		x+=2;
		pares_quadrados(n);
	}
	else return;
}

int main() {
	int num;
	scanf("%d", &num);
	pares_quadrados(num);
	return 0;
}
