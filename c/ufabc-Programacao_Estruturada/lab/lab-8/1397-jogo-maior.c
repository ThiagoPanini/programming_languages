#include <stdio.h>
#include <stdlib.h>
 
int main()
{
    int rept = 0, a = 0, b = 0, auxA = 0, auxB = 0;
    while (scanf("%d", &rept) && rept !=0){
        while (rept--){
        scanf("%d %d", &a, &b);
        if (a == b)
            auxA += 0;
        else if (a > b)
            auxA++;
        else if (a < b)
            auxB++;
    }
    printf("%d %d\n", auxA, auxB);
    auxA = 0;
    auxB = 0;
    }
    return 0;
}
