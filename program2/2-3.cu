#include <stdio.h>
#include <math.h>

// 各ブロックは16×16のスレッド
#define BLOCK 16
// 要素数N
#define N 65536

// カーネル関数
__global__ void Cal(int *Ad, int *Bd, int *Cd, int rootN)
{
	// 行(row)と列(column)
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	// Cの要素を計算
	Cd[rootN * r + c] = Ad[rootN * r + c] * Bd[N - 1 - (rootN * r + c)];
}

int main()
{
	// 使用するGPUの番号を決定
	cudaSetDevice(0);

	// i:繰り返し変数  size:要素数Nのサイズ
	int i, size = N * sizeof(int);
	// N個の要素を持つ動的配列A,B,C
	int *A, *Ad, *B, *Bd, *C, *Cd;

	// 計算時間計測のための変数
	float time;
	cudaEvent_t start = 0, stop = 0;

	// event変数startとstopを作成
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ホスト側の行列のメモリを確保
	A = (int*)malloc(size);
	B = (int*)malloc(size);
	C = (int*)malloc(size);

	// ホスト側の行列A,Bに要素を格納
	for (i = 0; i < N; i++)
	{
		A[i] = i;
		B[i] = i;
	}

	// デバイス側の行列のメモリを確保
	cudaMalloc(&Ad, size);
	cudaMalloc(&Bd, size);
	cudaMalloc(&Cd, size);

	// ホスト側の行列A,Bのデータをデバイス側の行列へ転送
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// グリッドとブロックの定義
	int rootN = int(sqrt(N));
	dim3 grid(rootN / BLOCK, rootN / BLOCK);
	dim3 block(BLOCK, BLOCK);

	printf("\nCalculation Start\n");
	// 開始時間を記録
	cudaEventRecord(start, 0);

	// GPU処理
	Cal <<<grid, block>>> (Ad, Bd, Cd, rootN);

	// 終了時間を記録
	cudaEventRecord(stop, 0);
	printf("\nCalculation End\n");

	// kernel終了までCPUを待機
	cudaEventSynchronize(stop);
	// stopとstartの時差をtimeに記録
	cudaEventElapsedTime(&time, start, stop);
	// 処理時間を表示
	printf("\nKernel time : %f [msec]\n", time);

	// デバイス側の計算結果をホストメモリー上の配列へコピー
	cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

	// nとCの計算結果を表示
	printf("\nif n = %d\n", N);
	for (int i = 0; i < N; i++)
	{
		printf("C(%d) = %d\n", i, C[i]);
	}

	// ホスト側のメモリ領域を解放
	free(A);
	free(B);
	free(C);

	// デバイス側のメモリ領域を解放
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(Cd);

	return 0;
}
