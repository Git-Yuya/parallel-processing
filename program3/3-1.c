#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    // 計算時間計測のための変数
    double start, end;
    // i,j,k:繰り返し変数  n:n行n列
    int i, j, k, n = pow(2, 4);
    // n個の要素を持つ動的配列A,B,HC,C
    // HC:アダマール積の計算結果  C:行列積の計算結果
    int **a, **b, **hc, **c;

    // コマンドライン引数(argv[0](実行ファイル) argv[1] argv[2] ...)
    // nをコマンドラインで変更する場合
    if (argc > 1)
    {
        n = atoi(argv[1]);
    }

    printf("matrix size = %d x %d\n", n, n);

    // メモリ領域の動的確保
    a = (int **)malloc(sizeof(int *) * n);
    b = (int **)malloc(sizeof(int *) * n);
    hc = (int **)malloc(sizeof(int *) * n);
    c = (int **)malloc(sizeof(int *) * n);
    for (i = 0; i < n; i++)
    {
        a[i] = (int *)malloc(sizeof(int) * n);
        b[i] = (int *)malloc(sizeof(int) * n);
        hc[i] = (int *)malloc(sizeof(int) * n);
        c[i] = (int *)malloc(sizeof(int) * n);
    }

    // A,Bの要素を格納
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            a[i][j] = 1;
            b[i][j] = 1;
        }
    }

    printf("\nCalculation Start\n");
    // 開始時間を記録
    start = omp_get_wtime();

    // HCとCの要素を計算して格納
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            // アダマール積の計算
            hc[i][j] = a[i][j] * b[i][j];

            // 行列積の計算
            c[i][j] = 0;
            for (k = 0; k < n; k++)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    // 終了時間を記録
    end = omp_get_wtime();
    printf("\nCalculation End\n");

    // HCとCの計算時間を表示
    printf("\nProcessing Time : %.10lf [s]\n", (double)(end - start));

    /*
    // nの値を表示
    printf("\nif n = %d\n", n);

    // HCの計算結果を表示
    printf("HC(Hadamard product) = \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%d ", hc[i][j]);
        }
        putchar('\n');
    }

    // Cの計算結果を表示
    printf("\nC(matrix product) = \n");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            printf("%d ", c[i][j]);
        }
        putchar('\n');
    }
    */

    // メモリ領域の解放
    for (i = 0; i < n; i++)
    {
        free(a[i]);
        free(b[i]);
        free(hc[i]);
        free(c[i]);
    }
    free(a);
    free(b);
    free(hc);
    free(c);

    return 0;
}
