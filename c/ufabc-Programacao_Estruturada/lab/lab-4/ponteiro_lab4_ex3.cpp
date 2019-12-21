#include <stdio.h>
int main () {
	printf (" -----------------------------\n");
	printf (" tipo\ttamanho ( bytes )\n");
	printf (" -----------------------------\n");
	printf (" char\t%u\n", sizeof ( char ));
	printf (" short\t%u\n", sizeof ( short ));
	printf (" int\t%u\n", sizeof ( int ));
	printf (" long\t%u\n", sizeof ( long ));
	printf (" long long\t%u\n", sizeof ( long long));
	printf (" long double\t%u\n", sizeof ( long double));
	printf (" -----------------------------\n");
	short *p1 = NULL ;
	printf ("* para short\t%u\n", sizeof (p1));
	printf ("conteudo de * para short\t%u\n", sizeof (* p1));
	float ** p2 = (float**)&p1; // ponteiro para ponteiro para float
	printf ("** para float\t%u\n", sizeof (p2));
	long double * p3 = (long double*)&p1; // ponteiro para long double
	printf ("* para long double\t%u\n", sizeof (p3));
	long double ** p4 = (long double**)&p1; // ponteiro para ponteiro para long double
	printf ("** para long double\t%u\n", sizeof (p4));		
	return 0;
}
