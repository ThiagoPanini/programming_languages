#include<stdio.h>

int num=0;

void sinal_paridade (int n) {
	int x;
	char par [] = "EVEN", impar [] = "ODD", neg [] = "NEGATIVE", pos [] = "POSITIVE", zero [] = "NULL";
	scanf("%d", &x);
	if(x==0)
		printf("%s\n", zero);
	else if (x%2==0)
		printf("%s ", par);
	else 
		printf("%s ", impar);
	if (x<0)
		printf("%s\n", neg);
	else if (x>0)
		printf("%s\n", pos);

	num++;
	if(num==n)
		return;
	sinal_paridade (n);
}

int main() {
	int N;
	scanf("%d", &N);
	sinal_paridade (N);
	
	return 0;
}

