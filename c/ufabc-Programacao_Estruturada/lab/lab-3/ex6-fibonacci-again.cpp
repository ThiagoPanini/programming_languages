#include <stdio.h>

long int fat(int n){
	 	
	if (n==0)
	 	return 1;
	else 
	 	return fat(n-1)*n;
	 }
int main (){
	
	int num,i=1,x=0;
	while (1){
		scanf("%d", &num);
		if (num==0) break;
		while (num>0){
		x += (num%10)*fat(i);
		num = num/10;
		i++;
		}
	printf("%d\n", x);
	i=1;
	x=0;
	}
	return 0;
}
