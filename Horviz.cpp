
#include "cuda_runtime.h"
#include <chrono>
#include <sstream>
#include <iostream>

#define arraySize 31 // 35 max


using namespace std;

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
  string line;
  float v;
  float* dev_coefs = new float[arraySize*2+1];
  while(getline(cin,line)){
    istringstream iss(line);
    int q = 0;
    while(iss>>v){
      dev_coefs[q] = v;
      q++;
}
int W = int(dev_coefs[arraySize*2]);
for(int i = 0; i < arraySize*2+1;i++){
  cout<<dev_coefs[i]<<" ";
}cout<<"\n";
  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  start = std::chrono::high_resolution_clock::now();

  float *additional_array = new float[arraySize];
  for (int i = 0; i < arraySize; i++) {
    additional_array[i] = dev_coefs[i + arraySize] / dev_coefs[i];
  }

  quickSortR(additional_array, dev_coefs, arraySize - 1);

  float t1, t2;
  float acceleration = 0;
  int *X = new int[arraySize];
  int *bestX = new int[arraySize];
  for (int i = 0; i < arraySize; i++) {
    X[i] = -1;
    bestX[i] = 0;
  }
  int curr_sum = 0;
  int reached_max = 0;
  float *cpu_bin = new float[arraySize];

  int h = 0;
  int k = h; // def_div;
  long int ns = 0;
  bool forward;
  while (h - k != -1) {
    ns++;
    forward = true;
    if (X[h] == -1) {
      X[h] = 1;
    } else {
      if (X[h] == 1) {
        X[h] = 0;
      } else {
        if (X[h] == 0) {
          X[h] = -1;
          h--;
          forward = false;
        }
      }
    }
    if (h == arraySize - 1) {
      int cw = 0;
      int cp = 0;
      for (int i = k; i < arraySize; i++) {
        cp += dev_coefs[i + arraySize] * X[i];
        cw += dev_coefs[i] * X[i];
      }
      if ((cw <= W) && (cp > reached_max)) {
        reached_max = cp;
        for (int i = k; i < arraySize; i++) {
          bestX[i] = X[i];
        }
      }
    } else {
      int cw = 0;
      for (int i = k; i < arraySize; i++) {
        cw += dev_coefs[i] * X[i];
      }
      if (cw > W)
        forward = false;
      cw = 0;
      float cp = 0;
      int nw = 0;
      int np = 0;
      for (int i = k; i < arraySize; i++) {
        np = X[i] != -1 ? X[i] * dev_coefs[i + arraySize]
                        : dev_coefs[i + arraySize];
        nw = X[i] != -1 ? X[i] * dev_coefs[i] : dev_coefs[i];
        if (cw + nw <= W) {
          cw += nw;
          cp += np;
        } else {
          cp += np * (W - cw) / nw;
          break;
        }
      }
      int b = cp;
      if (b <= reached_max) {
        forward = false;
      }
    }
    if (forward) {
      if (h < arraySize - 1) {
        h++;
      }
    }
  }

  end = std::chrono::high_resolution_clock::now();

  int elapsed_seconds =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "Время выполнения: " << elapsed_seconds << "microseconds\n";

  cout << "MAX = " << reached_max << "\n";
  for (int m = 0; m < arraySize; m++) {
    cout << bestX[m];
    curr_sum += bestX[m] * dev_coefs[m + arraySize];
  }
  cout << "\nЧисло итераций = " << ns << "\n";

}
  return 0;
}
