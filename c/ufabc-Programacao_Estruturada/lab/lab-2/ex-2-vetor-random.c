#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>

int main() {

	int i;
	double A[100], n;

	for(i=0; i<100; i++) {
		scanf("%lf", &n);
		A[i] = n;
	}
	
	for(i=0; i<100; i++) {
		if (A[i] <= 10) {
			printf("A[%d] = %.1lf\n", i, A[i]);
		}
	}

	return 0;
}
