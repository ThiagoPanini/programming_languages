#include<stdio.h>
int main()
{
    int N, i, s, x1, x2, aux;
    scanf("%d",&N);

    for(i = 1;i <= N;i++){
        scanf("%d %d", &x1,&x2);
        if(x1 < x2){
        aux=x1;
        x1=x2;
        x2=aux;
        }
        while(x1 % x2 != 0){
            s = x1;
            x1 = x2;
            x2 = s%x1;
        }
        printf("%d\n", x2);
    }
    return 0;
}
