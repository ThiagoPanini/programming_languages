#include <stdio.h>

int count=0;
long int vetorF[100]={0};

long int Fib(int n) { 
	count ++;
	
	if (n==0 || n==1)
		return n;
	if (vetorF[n]==0)
		vetorF[n] = Fib(n-1) + Fib(n-2);
		
	return vetorF[n];
}

int main() {
	int num;
	scanf("%d", &num);
	printf("%d\n", Fib(num));
	printf("%d\n", count);
	return 0;
}
