#include <stdio.h>

struct city {
	int pop;
	double cresc;
};

int main() {
	
	struct city A, B;
	int T, i, anos;
	scanf("%d", &T);
	
	for(i=1; i<=T; i++) {
		anos=0;
		scanf("%d %d %lf %lf", &(A.pop), &(B.pop), &(A.cresc), &(B.cresc));
		while (A.pop <= B.pop) {
			A.pop *= (A.cresc / 100.0) + 1.0;
			B.pop *= (B.cresc / 100.0) + 1.0;
			anos++;
			if (anos > 100) {
				printf("Mais de 1 seculo.\n");
				break;
			}
		}
		if (anos <= 100) {
			printf("%d anos.\n", anos);
		}
	}
	return 0;
}
