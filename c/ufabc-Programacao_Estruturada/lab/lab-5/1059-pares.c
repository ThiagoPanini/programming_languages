#include <stdio.h>

void pares(int n) {
	if (n == 0) {
		return;
	} else { 
		if ((102-n) % 2 == 0) {
			printf("%d\n", 102-n);
			pares(n-2);
		}
	}
}

int main() {
	pares(100);
	return 0;
}
