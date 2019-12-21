#include <stdio.h>
int main()
{
 long long int a, b, tmp, tmp2;
    int i;

 while(scanf("%lld %lld", &a, &b) != EOF)
 {
  tmp = 1;
  tmp2 = 1;
  for (i = a; i > 0; --i)
  {
   tmp *= a;
   a--;
  }

  for (i = b; i > 0; --i)
  {
   tmp2 *= b;
   b--;
  }

  printf("%lld\n", tmp + tmp2);
 }

 return 0;
}
