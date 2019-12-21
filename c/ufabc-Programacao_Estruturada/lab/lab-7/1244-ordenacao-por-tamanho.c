#include <stdio.h>

#define MAX_STRING 60
#define MAX_LEN 60

char strings[MAX_STRING][MAX_LEN];
int len_strings[MAX_STRING][2]; // 0 eh tamanho; 1 eh posicao
int aux[MAX_STRING][2]; // auxiliar de ordenacao

void merge_sort(int beg, int end);

int main()
{
    int n, num_strings, i;
    char c;

    scanf("%d%*c", &n);

    while(n--)
    {
        num_strings = len_strings[0][0] = len_strings[0][1] = 0;

        while(scanf("%c", &c) && c != '\n')
        {
            if( ('A' <= c && c <= 'Z') || ('a' <= c && c <= 'z') || ( '0' <= c && c <= '9') )
            {
                strings[num_strings][len_strings[num_strings][0]] = c;
                len_strings[num_strings][0]++;
            }
            else if(c == ' ')
            {
                strings[num_strings][len_strings[num_strings][0]] = 0;
                num_strings++;
                len_strings[num_strings][1] = num_strings;
                len_strings[num_strings][0] = 0;
            }
        }
        strings[num_strings][len_strings[num_strings][0]] = 0;

        merge_sort(0, num_strings);

        printf("%s", strings[len_strings[0][1]]);
        for(i = 1; i<=num_strings; i++)
        {
            printf(" %s", strings[len_strings[i][1]]);
        }
        printf("\n");
    }

    return 0;
}

void merge_sort(int beg, int end)
{
    if(beg == end)
        return;
    
    int i, j, len;

    merge_sort(beg, (beg+end)/2);
    merge_sort((beg+end)/2+1, end);

    for(len = 0, j = (beg+end)/2+1, i = beg; i <= (beg+end)/2; i++)
    {
        while(j <= end && len_strings[j][0] > len_strings[i][0])
        {
             aux[len][0] = len_strings[j][0];
             aux[len][1] = len_strings[j][1];
             len++;
             j++;
        }

        
        aux[len][0] = len_strings[i][0];
        aux[len][1] = len_strings[i][1];
        len++;
    }

    while(j <= end)
    {
            aux[len][0] = len_strings[j][0];
            aux[len][1] = len_strings[j][1];
            len++;
            j++;
    }

    for(i = beg; i <= end; i++)
    {
        len_strings[i][0] = aux[i-beg][0];
        len_strings[i][1] = aux[i-beg][1];
    }
}
