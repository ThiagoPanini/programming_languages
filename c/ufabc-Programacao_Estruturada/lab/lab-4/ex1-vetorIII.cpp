#include <stdio.h>
#include <stdlib.h>

int main() {
	
	// n é o ponteiro para double
	double *n;
	int i;
	
	n = (double*)malloc(100*sizeof(double));
	//n = malloc(100 * sizeof(double));
	
	scanf("%lf", n);
	
	for (i=0; i<100; i++) { 
		if(i>0) {
			*(n + i) = *(n+(i-1))/2;
		}
		printf("N[%d] = %.4lf\n", i, *(n + i));
	}
	
	return 0;
}
