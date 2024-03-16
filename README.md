# 並列処理
　C言語、C言語＋OpenMP、CUDAの3つの方法でプログラム1～3をそれぞれ実装し、計算時間を比較する。

## プログラムの説明
### プログラム1
　以下の計算を行うプログラムを実装し、 $L=2^7, L=2^{10}, L=2^{13}, \cdots$ と増やした時の計算時間を比較する。

$$ \text{SUM} = \sum_{i=0}^{L-1} \sum_{j=0}^{L-1} \hspace{2pt} (i^3 - j^3) $$

### プログラム2
　以下の計算を行うプログラムを実装し、 $n=2^7, n=2^{10}, n=2^{13}, \cdots$ と増やした時の計算時間を比較する。

$$ C(i) = A(i) \hspace{1pt} B(n - i) \quad (\text{ただし、}A(i) = B(i) = i, \hspace{1pt} i = 0, 1, \ldots, n - 1) $$

### プログラム3
　 $n$ 行 $n$ 列の要素1を持つ行列 $A$ と $B$ のアダマール積と行列積を求めるプログラムを実装し、 $n=2^7, n=2^{10}, n=2^{13}, \cdots$ と増やした時の計算時間を比較する。

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
