#include <stdio.h>

#include <string.h>

int main()
{
   char str[1000],str1[1000];
   int i,j,k,l,n,m,count,count1=0,T;
   scanf("%d",&T);

   while(T--)
   {
       scanf("%d",&l);
       getchar();
       count1=0;

       for(i=0;i<l;i++)
       {
            gets(str);
            k = strlen(str);

            for(n=0,count=0;n<k;n++)
            {
                j = (int)(str[n]-64);
                count = count+j-1+n+i;
            }

            count1 = count1 + count;
       }
       printf("%d\n",count1);
   }
}
