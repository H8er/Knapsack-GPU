
#include "cuda_runtime.h"
#include <chrono>
#include <iostream>
#include <sstream>

#define arraySize 31 // 35 max
#define def_div 10   // 5<=X<=15
//#define W 100
//#define threads_per_block 32
//#define max_blocks 32

using namespace std;

__constant__ float coefs[arraySize * 2 + 1];
__global__ void hybrid(float *sh_sum_dev, long int *str_num_dev,
                       float num_of_blocks, int *bdevX, int *global_mem_bin,
                       int threads_per_block) {
  float th_w_sum = 0;
  float th_v_sum = 0;
  int th_bin[arraySize];
  int best_bin[arraySize];
  extern __shared__ float sh_array[];
  float *sh_maxs = (float *)sh_array;
  int *indices = (int *)&sh_maxs[threads_per_block];
  int reached = 0;
  indices[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();

  long signed int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
// num_to_bin += max_blocks * n_of_it;
#pragma unroll
  for (uint i = 0; i < def_div; i++) {
    th_bin[i] = ((num_to_bin) >> i) % 2;
    th_w_sum += th_bin[i] * coefs[i];
    th_v_sum += th_bin[i] * coefs[i + arraySize];
    best_bin[i] = th_bin[i];
  }
#pragma unroll
  for (uint i = def_div; i < arraySize; i++) {
    th_bin[i] = -1;
  }
  int Capacity = coefs[arraySize * 2] - th_w_sum;
  sh_maxs[threadIdx.x] = (th_w_sum > coefs[arraySize * 2]) ? 0 : th_v_sum;
  __syncthreads();

  // H_S
  int h = def_div;
  long int ns = 0;
  bool forward;

  while (h - def_div != -1) {
    ns++;
    forward = true;
    if (th_bin[h] == -1) {
      th_bin[h] = 1;
    } else {
      if (th_bin[h] == 1) {
        th_bin[h] = 0;
      } else {
        if (th_bin[h] == 0) {
          th_bin[h] = -1;
          h--;
          forward = false;
        }
      }
    }
    if (h == arraySize - 1) {
      int cw = 0;
      int cp = 0;
#pragma unroll
      for (int i = def_div; i < arraySize; i++) {
        cp += coefs[i + arraySize] * th_bin[i];
        cw += coefs[i] * th_bin[i];
      }
      if ((cw <= Capacity) && (cp > reached)) {
        reached = cp;
#pragma unroll
        for (int i = def_div; i < arraySize; i++) {
          best_bin[i] = th_bin[i];
        }
      }
    } else {
      int cw = 0;
      for (int i = def_div; i < arraySize; i++) {
        cw += coefs[i] * th_bin[i];
      }
      if (cw > Capacity)
        forward = false;
      cw = 0;
      float cp = 0;
      int nw = 0;
      int np = 0;
#pragma unroll
      for (int i = def_div; i < arraySize; i++) {
        np = th_bin[i] != -1 ? th_bin[i] * coefs[i + arraySize]
                             : coefs[i + arraySize];
        nw = th_bin[i] != -1 ? th_bin[i] * coefs[i] : coefs[i];
        if (cw + nw <= Capacity) {
          cw += nw;
          cp += np;
        } else {
          cp += np * (Capacity - cw) / nw;
          break;
        }
      }
      int b = cp;
      if (b <= reached) {
        forward = false;
      }
    }
    if (forward) {
      if (h < arraySize - 1) {
        h++;
      }
    }
  }

  sh_maxs[threadIdx.x] += reached;

  __syncthreads();
  // reduction on block
  for (uint offset = blockDim.x >> 1; offset >= 1; offset >>= 1) {
    if (threadIdx.x < offset) {
      if (sh_maxs[threadIdx.x] < sh_maxs[threadIdx.x + offset]) {
        sh_maxs[threadIdx.x] = sh_maxs[threadIdx.x + offset];
        indices[threadIdx.x] = indices[threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (threadIdx.x == 0) {
    sh_sum_dev[blockIdx.x] = sh_maxs[0];
    str_num_dev[blockIdx.x] = indices[0];
  }
  if (blockIdx.x * blockDim.x + threadIdx.x == indices[0]) {
#pragma unroll
    for (int i = 0; i < arraySize; i++) {
      global_mem_bin[blockIdx.x * arraySize + i] = best_bin[i];
    }
  }
  __syncthreads();
}

__global__ void hybrid_reduction(float *s, long int *str_num_dev,
                                 int *global_mem_bin, int threads_per_block) {
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int sh_hy_data[];
  sh_hy_data[threadIdx.x] = s[ID];
  sh_hy_data[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads();
  // do reduction in shared mem
  for (uint s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      if (sh_hy_data[threadIdx.x] < sh_hy_data[threadIdx.x + s]) {
        sh_hy_data[threadIdx.x] = sh_hy_data[threadIdx.x + s];
        sh_hy_data[threadIdx.x + threads_per_block] =
            sh_hy_data[threadIdx.x + threads_per_block + s];
      }
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (threadIdx.x == 0) {
    // if(sh_hy_data[0]>s[0]){//}&&(blockIdx.x>0)){
    s[blockIdx.x] = sh_hy_data[0];
    str_num_dev[blockIdx.x] = sh_hy_data[threads_per_block];

#pragma unroll
    for (int i = 0; i < arraySize; i++) {
      global_mem_bin[i] =
          global_mem_bin[(sh_hy_data[threads_per_block] / threads_per_block) *
                             arraySize +
                         i];
    }
  }
}

__global__ void which_string(long int a, int *view_dev) {
  view_dev[threadIdx.x] = (a >> threadIdx.x) % 2;
}

void quickSortR(float *a, float *b, long N) {
  // На входе - массив a[], a[N] - его последний элемент.

  long i = 0, j = N; // поставить указатели на исходные места
  float temp, p;

  p = a[N >> 1]; // центральный элемент

  // процедура разделения
  do {
    while (a[i] > p)
      i++;
    while (a[j] < p)
      j--;

    if (i <= j) {
      temp = a[i];
      a[i] = a[j];
      a[j] = temp;
      temp = b[i];
      b[i] = b[j];
      b[j] = temp;
      temp = b[i + arraySize];
      b[i + arraySize] = b[j + arraySize];
      b[j + arraySize] = temp;
      i++;
      j--;
    }
  } while (i <= j);

  // рекурсивные вызовы, если есть, что сортировать
  if (j > 0)
    quickSortR(a, b, j);
  if (N > i)
    quickSortR(a + i, b + i, N - i);
}

int main() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int threads_per_block = deviceProp.warpSize;
  int max_blocks = pow(2, def_div) / threads_per_block;
  long int strSize_b = pow(2, arraySize);
  int num_of_blocks = strSize_b / threads_per_block;
  float *Sum = new float[32]; // = { 0 };
  float *sh_sum_dev;

  int iter = 0;

  string line;
  float v;
  float *dev_coefs = new float[arraySize * 2 + 1];
  while (getline(cin, line)) {
    istringstream iss(line);
    int q = 0;
    while (iss >> v) {
      dev_coefs[q] = v;
      q++;
    }
    cout<<"Iter = "<<iter<<"\n";iter++;
    for (int i = 0; i < arraySize * 2 + 1; i++) {
      cout << dev_coefs[i] << " ";
    }
    cout << "\n";
    // int W = int(dev_coefs[arraySize*2]);

    long int *str_num_dev;
    long int *str_num = new long int[1];

    float *additional_array = new float[arraySize];
    for (int i = 0; i < arraySize; i++) {
      additional_array[i] = dev_coefs[i + arraySize] / dev_coefs[i];
    }

    quickSortR(additional_array, dev_coefs, arraySize - 1);

    int *bdevX;
    cudaMalloc((void **)&bdevX, arraySize * sizeof(int));
    int *global_mem_bin;
    cudaMalloc((void **)&global_mem_bin, max_blocks * arraySize * sizeof(int));

    cudaMalloc((void **)&sh_sum_dev, num_of_blocks * sizeof(float));
    cudaMalloc((void **)&str_num_dev, num_of_blocks * sizeof(long));
    cudaMemcpyToSymbol(coefs, dev_coefs, (2 * arraySize + 1) * sizeof(float));

    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();



    hybrid<<<max_blocks, threads_per_block,
             threads_per_block * 3 * sizeof(int)>>>(
        sh_sum_dev, str_num_dev, num_of_blocks, bdevX, global_mem_bin,
        threads_per_block);

    hybrid_reduction<<<1, max_blocks, threads_per_block * 3 * sizeof(int)>>>(
        sh_sum_dev, str_num_dev, global_mem_bin, threads_per_block);

    int *suda = new int[arraySize];
    cudaMemcpy(Sum, sh_sum_dev, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(str_num, str_num_dev, sizeof(long int), cudaMemcpyDeviceToHost);
    cudaMemcpy(suda, global_mem_bin, arraySize * sizeof(int),
               cudaMemcpyDeviceToHost);

    end = std::chrono::high_resolution_clock::now();

    int elapsed_seconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    std::cout << "Время выполнения: " << elapsed_seconds << "microseconds\n";
    cout << "Acheived maximal sum = " << Sum[0] << "\n";
    cout << str_num[0] << "\n";
    for (int i = 0; i < arraySize; i++) {
      cout << suda[i];
    }
    cout << "\n";

    // check
    int checksum = 0;
    for (int i = 0; i < arraySize; i++) {
      checksum += dev_coefs[i + arraySize] * suda[i];
    }
    cout << "Validation sum = " << checksum << "\n";
    checksum = 0;
    for (int i = 0; i < arraySize; i++) {
      checksum += dev_coefs[i] * suda[i];
    }
    cout << "Weight = " << checksum << "\n";

    cudaFree(coefs);
    cudaFree(sh_sum_dev);
    cudaFree(str_num_dev);
    cudaFree(bdevX);
    cudaFree(global_mem_bin);
/*
    delete[] Sum;
    delete[] suda;
    delete[] str_num;
    delete[] dev_coefs;
    delete[] additional_array;
*/
    //cudaDeviceReset();
  }
  return 0;
}
