#include <stdio.h>
#include <stdlib.h>

int main() { 
	
	double *x, *y;
	x = (double *)malloc(1*sizeof(double));
	y = (double *)malloc(1*sizeof(double));
	
	scanf("%lf %lf", x, y);
	printf("%.3lf km/l\n", (*x)/(*y));
	
	return 0;
}
