#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

int main(int argc, char* argv[])
{
    // 計算時間計測のための変数
    double start, end;
    // i,j:繰り返し変数  L:繰り返し回数
    int i, j, L = pow(2, 8);
    // sum:計算結果の和
    long long sum = 0;

    // コマンドライン引数(argv[0](実行ファイル) argv[1] argv[2] ...)
    // 繰り返し回数をコマンドラインで変更する場合
    if (argc > 1)
    {
        L = atoi(argv[1]);
    }

    printf("\nCalculation Start\n");
    // 開始時間を記録
    start = omp_get_wtime();
    
    // SUMの計算
    for (i = 0; i < L; i++)
    {
        for (j = 0; j < L; j++)
        {
            // sum = sum + i^3 - j^3
            sum += (i * i * i) - (j * j * j);
        }
    }
    
    // 終了時間を記録
    end = omp_get_wtime();
    printf("\nCalculation End\n");

    // SUMの計算時間を表示
    printf("\nProcessing Time : %.10lf [s]\n", (double)(end - start));

    // LとSUMの計算結果を表示
    printf("\nL = %d\nSUM = %lld\n", L, sum);
    
    return 0;
}
