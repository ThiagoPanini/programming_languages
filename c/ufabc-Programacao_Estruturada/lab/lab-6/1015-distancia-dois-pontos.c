#include <stdio.h>
#include <math.h>

struct dist_pontos {
	float x;
	float y;
};

int main() {
	
	struct dist_pontos m, n;
	scanf("%f %f", &(m.x), &(m.y));
	scanf("%f %f", &(n.x), &(n.y));
	
	printf("%.4f\n", sqrt((pow((m.x - n.x), 2) + pow((m.y - n.y), 2))));
	
	return 0;
}
