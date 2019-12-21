#include<stdio.h>

int main() {
	
	int Y, X, i, j, k, l=0;
	scanf("%d", &Y);
	
	for(i=1; i<=Y; i++) {
		scanf("%d", &X);
		k=X/2;
		l=0;
		for(j=1; j<=k; j++) {
			if(X%j==0) {
				l+=j;
			}
		}
		if(l==X) {
			printf("%d eh perfeito\n", X);
		} else {
			printf("%d nao eh perfeito\n", X);
		}
	}
	
	return 0;
}
