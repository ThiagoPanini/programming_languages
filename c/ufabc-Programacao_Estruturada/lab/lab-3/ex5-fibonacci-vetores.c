#include<stdio.h>
int main()
{
   long long int n, first= 0, sec = 1, prox, i;
   int j,k;
   scanf("%d", &k);
   for(j=1; j<=k; j++, first=0, sec=1)
   {
       scanf("%lld",&n);
       n=n+1;
       for (i=0; i<n; i++)
       {
          if (i<=1) prox=i;
          else
          {
             prox = first + sec;
             first = sec;
             sec = prox;
          }
       }
          printf("Fib(%lld) = %lld\n",n-1,prox);
   }
   return 0;
}
