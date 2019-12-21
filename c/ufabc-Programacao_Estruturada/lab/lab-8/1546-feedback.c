#include <stdio.h>
 
int main() {
 
 int p = 0, q = 0, n, k, x;
 
 scanf("%d", &n);
 
 for(p = 0; p < n; ++p)
 {
  scanf("%d", &k);
  
  for(q = 0; q < k; ++q)
  {
   scanf("%d", &x);
   
   if(x == 1){
    printf("Rolien\n");
   }else if(x == 2){
       printf("Naej\n");
   }else if(x == 3){
       printf("Elehcim\n");
   }else{
       printf("Odranoel\n");
   }
  }
 }
 
    return 0;
}
