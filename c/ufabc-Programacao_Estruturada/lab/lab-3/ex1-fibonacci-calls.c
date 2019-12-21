#include <stdio.h>

int count=0;

long int Fib(int n) {
	count++;
	if (n==0 || n==1)
		return n;
	else
		return Fib(n-1) + Fib(n-2);
}

int main() {
	int num, repeat, i, result;
	
	scanf("%d", &repeat);
	for(i=0; i<repeat; i++){
		scanf("%d", &num);
		result = Fib(num);
		printf("fib(%d) = %d calls = %d\n", num, count-1, result);
		count=0;
	}
	
	
	return 0;
}
