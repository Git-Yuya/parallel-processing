#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    // 計算時間計測のための変数
    double start, end;
    // i,j:繰り返し変数  n:要素数
    int i, j, n = pow(2, 12);
    // n個の要素を持つ動的配列A,B,C
    int *a, *b, *c;

    // コマンドライン引数(argv[0](実行ファイル) argv[1] argv[2] ...)
    // 要素数をコマンドラインで変更する場合
    if (argc > 1)
    {
        n = atoi(argv[1]);
    }

    // メモリ領域の動的確保
    a = (int*)malloc(sizeof(int) * n);
    b = (int*)malloc(sizeof(int) * n);
    c = (int*)malloc(sizeof(int) * n);

    // A,Bの要素を格納
    for (i = 0; i < n; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    printf("\nCalculation Start\n");
    // 開始時間を記録
    start = omp_get_wtime();

    // Cの要素を計算して格納
    for (i = 0; i < n; i++)
    {
        c[i] = a[i] * b[n - 1 - i];
    }
    
    // 終了時間を記録
    end = omp_get_wtime();
    printf("\nCalculation End\n");

    // Cの計算時間を表示
    printf("\nProcessing Time : %.10lf [s]\n", (double)(end - start));

    // nとCの計算結果を表示
    printf("\nif n = %d\n", n);
    for (int i = 0; i < n; i++)
    {
        printf("C(%d) = %d\n", i, c[i]);
    }

    // メモリ領域の解放
    free(a);
    free(b);
    free(c);
    
    return 0;
}
