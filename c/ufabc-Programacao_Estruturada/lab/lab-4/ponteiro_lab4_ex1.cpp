# include <stdio.h>
int main () {
	int x1 = 2006; // inteiro x1 recebe 2006
	int *p1 = &x1; // ponteiro p1 recebe endereco de x1
	int x2 = 2017; // inteiro x2 recebe 2017
	int *p2 = &x2; // ponteiro p2 recebe endereco de x2
	char c = 'u'; // caractere c recebe 'u'
	printf ("%p\n", &c); // imprime um endereco ( algo como 0 x7ffffe0c697f )
	printf ("%p\n", &x1); // imprime um endereco ( algo como 0 x7ffffe0c6980 )
	printf ("%p\n", &x2); // imprime um endereco ( algo como 0 x7ffffe0c6984 )
	printf ("%p\n", p1);
	printf ("%p\n", p2);
	printf ("%d\n", *p1 + *p2);
	return 0;
}
