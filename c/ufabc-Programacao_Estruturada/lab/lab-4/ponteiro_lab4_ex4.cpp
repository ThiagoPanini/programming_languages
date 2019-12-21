# include <stdio.h>
# include <stdlib.h>
int main () {
	int n = 10;
	printf ("n = ");
	scanf ("%d", &n);
	int *v = (int*)malloc(n*sizeof(int));
	for ( int i = 0; i < n; i ++)
	v[i] = i;
	printf ("n [0]: %d\n", *(v));
	printf ("n [0]: %d\n", v [0]) ;
	printf ("n[n -1]: %d\n", *(v + n - 1));
	printf ("n[n -1]: %d\n", v[n - 1]) ;
	free (v);
	v = NULL ;
	return 0;
}
