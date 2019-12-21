#include <stdio.h>


int main() {
	int arr[10];
	int i, j;
	
	for (i=0; i<10; i++) {
		scanf("%d", &arr[i]);
		if (arr[i] < 1) {
			arr[i] = 1;
		}
	}
	
	for (j=0; j<10; j++) {
		printf("X[%d] = %d\n", j, arr[j]);
	}	
	
	return 0;
}
