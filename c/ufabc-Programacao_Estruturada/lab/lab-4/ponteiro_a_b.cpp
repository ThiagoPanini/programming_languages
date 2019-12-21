#include<stdio.h>

void troca(int *i, int *j) {
	int temp;
	temp = *i;
	*i = *j;
	*j = temp;
}

void troca_fake(int *i, int *j) {
	int *temp = NULL;
	temp = i;
	*i = *j;
	*j = *temp;
}

int main() {
	int a=1;
	int b=10;
	
	troca(&a,&b);
	printf("\ta=%d\n\tb=%d", a, b);
	troca_fake(&a,&b);
	printf("\n\n\ta=%d\n\tb=%d", a, b);	
}
