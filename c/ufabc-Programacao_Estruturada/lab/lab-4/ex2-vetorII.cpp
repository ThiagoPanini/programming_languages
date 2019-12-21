#include <stdio.h>
#include <stdlib.h>

int main() {
	
	int t, i, *temp, *n;
	
	n = (int *) malloc(1000 * sizeof(int));
	temp = (int *) malloc(1 * sizeof(int));
	
	*temp = 0;
	
	scanf("%d", &t);
	
	for (i=0; i<1000; i++){
		if(*temp != t) {
			*(n+i) = *temp;
			*temp = *temp + 1;
		}
		if (*temp == t) {
			*temp = 0;
			
		}
		printf("N[%d] = %d\n", i, n[i]);
	}
	
		
	return 0;
}
