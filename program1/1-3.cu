#include <stdio.h>

// 各ブロックは16×16のスレッド
#define BLOCK 16
// 繰り返し回数L
#define L 256

// カーネル関数
__global__ void Sum(int *Ad, long long *sumd)
{
    // 行(row)と列(column)
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    // 加算
    *sumd += Ad[L * r + c];
}

int main()
{
    // 使用するGPUの番号を決定
    cudaSetDevice(0);

    // i,j:繰り返し変数  size:L×Lのサイズ
    int i, j, size = L * L * sizeof(int);
    // L*Lの要素を持つ動的配列A,Ad
    int *A, *Ad;
    // 和を格納する変数sum,sumd
    long long sum = 0, sumd = 0;

    // 計算時間計測のための変数
    float time;
    cudaEvent_t start, stop;

    // event変数startとstopを作成
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ホスト側の行列のメモリを確保
    A = (int*)malloc(size);

    // ホスト側の行列Aに要素を格納
    for (i = 0; i < L; i++)
    {
        for (j = 0; j < L; j++)
        {
            A[L * i + j] = (i * i * i) - (j * j * j);
        }
    }

    // デバイス側の行列のメモリを確保
    cudaMalloc(&Ad, size);

    // ホスト側の行列Aとsumのデータをデバイス側の行列へ転送
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(&sumd, &sum, sizeof(sum), cudaMemcpyHostToDevice);

    // グリッドとブロックの定義
    dim3 grid(L / BLOCK, L / BLOCK);
    dim3 block(BLOCK, BLOCK);

    printf("\nCalculation Start\n");
    // 開始時間を記録
    cudaEventRecord(start, 0);

    // GPU処理
    Sum <<<grid, block>>> (Ad, &sumd);

    // 終了時間を記録
    cudaEventRecord(stop, 0);
    printf("\nCalculation End\n");

    // kernel終了までCPUを待機
    cudaEventSynchronize(stop);
    // stopとstartの時差をtimeに記録
    cudaEventElapsedTime(&time, start, stop);
    // 処理時間を表示
    printf("\nKernel time : %f [msec]\n", time);

    // デバイス側の計算結果をsumへコピー
    cudaMemcpy(&sum, &sumd, sizeof(sum), cudaMemcpyDeviceToHost);

    // LとSUMの計算結果を表示
    printf("\nL = %d\nSUM = %lld\n", L, sum);

    // ホスト側のメモリ領域を解放
    free(A);

    // デバイス側のメモリ領域を解放
    cudaFree(Ad);

    return 0;
}
