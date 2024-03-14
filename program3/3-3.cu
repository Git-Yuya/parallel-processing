#include <stdio.h>

// 各ブロックは16×16のスレッド
#define BLOCK 16
// 行列N×N
#define N 16

// カーネル関数
__global__ void MXProduct(int *Ad, int *Bd, int *HCd, int *Cd)
{
	// 行(row)と列(column)
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	// アダマール積
	HCd[N * r + c] = Ad[N * r + c] * Bd[N * r + c];

	// 行列積
	int tmp = 0;
	for (int i = 0; i < N; i++)
	{
		tmp += Ad[N * r + i] * Bd[N * i + c];
	}
	Cd[N * r + c] = tmp;
}

int main()
{
	// 使用するGPUの番号を決定
	cudaSetDevice(0);

	// i,j:繰り返し変数  size:N行N列のサイズ
	int i, j, size = N * N * sizeof(int);
	// N*N個の要素を持つ動的配列A,B,HC,C
	// HC:アダマール積の計算結果  C:行列積の計算結果
	int *A, *Ad, *B, *Bd, *HC, *HCd, *C, *Cd;

	// 計算時間計測のための変数
	float time;
	cudaEvent_t start, stop;
	// event変数startとstopを作成
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ホスト側の行列のメモリを確保
	A = (int*)malloc(size);
	B = (int*)malloc(size);
	HC = (int*)malloc(size);
	C = (int*)malloc(size);

	// ホスト側の行列A,Bに要素を格納
	for (i = 0; i < N * N; i++)
	{
		A[i] = 1;
		B[i] = 1;
	}

	// デバイス側の行列のメモリを確保
	cudaMalloc(&Ad, size);
	cudaMalloc(&Bd, size);
	cudaMalloc(&HCd, size);
	cudaMalloc(&Cd, size);

	// ホスト側の行列A,Bのデータをデバイス側の行列へ転送
	cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

	// グリッドとブロックの定義
	dim3 grid(N / BLOCK, N / BLOCK);
	dim3 block(BLOCK, BLOCK);

	printf("\nCalculation Start\n");
	// 開始時間を記録
	cudaEventRecord(start, 0);

	// GPU処理
	MXProduct <<<grid, block>>> (Ad, Bd, HCd, Cd);

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
	cudaMemcpy(HC, HCd, size, cudaMemcpyDeviceToHost);

	// nの値を表示
	printf("\nif n = %d\n", N);

	// HCの計算結果を表示
	printf("HC(Hadamard product) = \n");
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%d ", HC[i * N + j]);
		}
		putchar('\n');
	}

	// Cの計算結果を表示
	printf("\nC(matrix product) = \n");
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < N; j++)
		{
			printf("%d ", C[i * N + j]);
		}
		putchar('\n');
	}

	// ホスト側のメモリ領域を解放
	free(A);
	free(B);
	free(HC);
	free(C);

	// デバイス側のメモリ領域を解放
	cudaFree(Ad);
	cudaFree(Bd);
	cudaFree(HCd);
	cudaFree(Cd);

	return 0;
}
