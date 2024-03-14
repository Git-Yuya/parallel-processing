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
    
    // OpenMPによる並列化
    #pragma omp parallel /* num_threads() */
    {
        // SUMの計算
        // nowaitによって、処理が終了したスレッドは他のスレッドの状況に関係なく次の処理に移行
        // iとjはプライベート変数
        #pragma omp for nowait private(j)
        for (i = 0; i < L; i++)
        {
            for (j = 0; j < L; j++)
            {
                // 他のスレッドと同時に実行されないようにcritical領域を指定
                #pragma omp critical
                {
                    // sum = sum + i^3 - j^3
                    sum += (i * i * i) - (j * j * j);
                }
            }
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
