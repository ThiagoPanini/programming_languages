#include <stdio.h>

int main() {
	
	double N[100], num;
	int i;
	
	scanf("%lf",&num);
	N[0]=num;
	printf("N[0] = %.4lf\n", N[0]);
	
	for (i=1; i<100; i++){
		N[i]=N[i-1]/2;
		printf("N[%d] = %.4lf\n", i, N[i]);
		if (N[i]==0.0000){
			break;
		}
	}
	return 0;
}
