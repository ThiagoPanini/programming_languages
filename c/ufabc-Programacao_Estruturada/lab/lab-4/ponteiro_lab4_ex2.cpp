# include <stdio.h>
int main () {
	float *p3 = NULL ;
	float x3 = 100.0;
	p3 = &x3;
	printf ("%f\n", *p3); // imprime o conteudo apontado por p3
	*p3 = 10.5;
	printf ("%f\n", *p3); // imprime o conteudo apontado por p3
	printf ("%f\n", x3); // imprime o valor de x3
	printf ("%p\n", &p3); // imprime o endereco do ponteiro p3
	printf ("%p\n", p3); // imprime o endereco apontado pelo ponteiro p3
	printf ("%p\n", &x3); // imprime o endereco de x3
	return 0;
}
