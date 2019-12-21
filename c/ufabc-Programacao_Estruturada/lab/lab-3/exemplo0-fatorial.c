#include <stdio.h>

int fatorial(int n) {
	if (n==0)
		return 1;
	else 
		return fatorial(n-1)*n;
}

int main() {
	int num1, num2, soma;
	
	scanf("%d %d", &num1, &num2);
	soma = fatorial(num1) + fatorial(num2);
	printf("%d\n", soma);
	
	return 0;
}
