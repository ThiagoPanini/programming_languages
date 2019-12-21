#include <stdio.h>

int eh_div(int n, int i) {
	if (i <= n) {
		if (n % i == 0) {
			printf("%d\n", i);
		}
		eh_div(n, i+1);
	}
}

int main() {
	int n;
	scanf("%d", &n);
	eh_div(n, 1);
	
	return 0;
}
