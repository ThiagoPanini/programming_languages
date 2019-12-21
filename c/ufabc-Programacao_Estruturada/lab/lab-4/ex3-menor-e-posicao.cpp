#include <stdio.h>
#include <stdlib.h>

int main() {
	
	int n, i, j, menor=0, p;
	int *x;
	
	scanf("%d", &n);
	
	x = (int *)malloc(n * sizeof(int));
	
	for (i=0; i<n; i++) {
		scanf("%d", (x+i));
		for(j=i; j>=0; j--) {
			if(*(x+i) < menor) {
				menor=*(x+i);
				p = i;
			}
		}
	}
	
	printf("Menor valor: %d\nPosicao: %d\n", menor, p);
	
	
	return 0;
}
