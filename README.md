# 並列処理
　C言語、C言語＋OpenMP、CUDAの3つの方法でプログラム1～3をそれぞれ実装し、計算時間を比較・評価する。

## 評価
### プログラム1
　以下の計算を行うプログラムを実装し、 $L=2^8, L=2^{12}, \cdots$ と増やした時の計算時間を比較する。

$$ \text{SUM} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \hspace{2pt} (i^3 - j^3) $$

<br>

- $L=256(=2^8)$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0001194 | 100 |
| C言語＋OpenMP | 0.0065664 | 5499 |
| CUDA | 0.0000000 | 0 |

<br>

- $L=4096(=2^{12})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0281862 | 100 |
| C言語＋OpenMP | 2.0924867 | 7424 |
| CUDA | 0.0000000 | 0 |

<br>

- $L=65536(=2^{16})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 6.9393366 | 100 |
| C言語＋OpenMP | 563.89854 | 8126 |
| CUDA | メモリ不足 | - |

### プログラム2
　以下の計算を行うプログラムを実装し、 $n=2^{12}, n=2^{16}, \cdots$ と増やした時の計算時間を比較する。

$$ C(i) = A(i) \hspace{1pt} B(n - i) \quad (\text{ただし、}A(i) = B(i) = i, \hspace{1pt} i = 0, 1, \ldots, n - 1) $$

<br>

- $L=4096(=2^{12})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0000118 | 100 |
| C言語＋OpenMP | 0.0005082 | 4307 |
| CUDA | 0.000007168 | 60.75 |

<br>

- $L=65536(=2^{16})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0002023 | 100 |
| C言語＋OpenMP | 0.0008619 | 4261 |
| CUDA | 0.000017408 | 2.02 |

<br>

- $L=1048576(=2^{20})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0027411 | 100 |
| C言語＋OpenMP | 0.0017855 | 65.14 |
| CUDA | 0.000185344 | 6.76 |

<br>

- $L=16777216(=2^{24})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0395946 | 100 |
| C言語＋OpenMP | 0.0216864 | 54.77 |
| CUDA | 0.002864128 | 7.23 |

### プログラム3
　 $n$ 行 $n$ 列の要素1を持つ行列 $A$ と $B$ のアダマール積と行列積を求めるプログラムを実装し、 $n=2^4, n=2^8, \cdots$ と増やした時の計算時間を比較する。

<br>

- $L=16(=2^{4})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0000136 | 100 |
| C言語＋OpenMP | 0.0012649 | 9301 |
| CUDA | 0.000008192 | 60.24 |

<br>

- $L=256(=2^{8})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 0.0556386 | 100 |
| C言語＋OpenMP | 0.0171484 | 30.82 |
| CUDA | 0.000607232 | 1.091 |

<br>

- $L=1048576(=2^{20})$ の時の計算時間

| 方法 | 計算時間 [s] | 加速化率 [%] |
|:---:|:---:|:---:|
| C言語 | 1570.3204803 | 100 |
| C言語＋OpenMP | 471.3444349 | 30.02 |
| CUDA | 2.625450928 | 0.1672 |


## 各ファイルのコンパイル方法
- C言語
  ```
  cl filename.c
  ```
- C言語＋OpenMP
  ```
  cl filename.c -openmp
  ```
- CUDA
  ```
  nvcc filename.cu -o filename.exe
  ```

## 実装
- 言語：
  <img src="https://img.shields.io/badge/-C%E8%A8%80%E8%AA%9E-A8B9CC.svg?logo=c&style=plastic">
  <img src="https://img.shields.io/badge/-CUDA-76B900.svg?logo=nvidia&style=plastic">
- 統合開発環境：
  <img src="https://img.shields.io/badge/-Visual%20Studio-5C2D91.svg?logo=visualstudio&style=plastic">
