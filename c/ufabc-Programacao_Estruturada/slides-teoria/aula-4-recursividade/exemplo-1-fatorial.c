#include <stdio.h>

int fatorial(int n) {
	if (n==0) {
		return 1;
	}
	else {
		return fatorial(n-1)*n;
	}
}

int main() {
	int num;
	
	scanf("%d\n", &num);
	printf("%d\n", fatorial(num));
	
	return 0;
}
