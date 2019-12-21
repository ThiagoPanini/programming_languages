#include <stdio.h>

int main() {
	
	struct color {
		int R;
		int G;
		int B;
	};
	
	int MAX(int x, int y, int z) {
		int aux = x;
		if(aux < y){aux = y;}
		if(aux < z){aux = z;}
		return aux;
	}
	
	int MIN(int x, int y, int z) {
		int aux = x;
		if(aux > y){aux = y;}
		if(aux > z){aux = z;}
		return aux;
	}
	
	struct color rgb;
	int T, i, P=0, min, max;
	char conversor[5];
	scanf("%d", &T);
	
	for(i=1; i<=T; i++) {
		scanf("%s %d %d %d", conversor, &(rgb.R), &(rgb.G), &(rgb.B));
		
		if(conversor[1] == 'y') {
			P = 0.3*rgb.R + 0.59*rgb.G + 0.11*rgb.B;
		}
		else if (conversor[1] == 'e') {
			P = (rgb.R + rgb.G + rgb.B) / 3;
		}
		else if (conversor[1] == 'i') {
			P = MIN(rgb.R, rgb.G, rgb.B);
		}
		else if (conversor[1] == 'a') {
			P = MAX(rgb.R, rgb.G, rgb.B);
		}
		printf("Caso #%d: %d\n", i, P);
	}
	return 0;
}
