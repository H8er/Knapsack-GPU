
#include <chrono>
#include <iostream>
#include <math.h>
#include <sstream>

#define arraySize 31 // 35 max

using namespace std;

int main() {
  string line;
  float v;
  long int strSize_b = pow(2, arraySize);
  float *dev_coefs = new float[arraySize * 2 + 1];
  while (getline(cin, line)) {
    istringstream iss(line);
    int q = 0;
    while (iss >> v) {
      dev_coefs[q] = v;
      q++;
    }
    int W = int(dev_coefs[arraySize * 2]);
    float *cpu_bin = new float[arraySize];
    int max = 0;
    int cpu_str = 0;
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    for (long int i = 0; i < strSize_b; i++) {
      int tmp = 0;
      int cap = 0;
      long int tobin = i;
      for (int k = 0; k < arraySize; k++) {
        cpu_bin[k] = tobin % 2;
        tobin >>= 1;
        tmp += cpu_bin[k] * dev_coefs[arraySize + k]; //
        cap += cpu_bin[k] * dev_coefs[k];
      }
      if ((cap <= W) && (tmp > max)) {
        max = tmp;
        cpu_str = i;
      }
    }

    end = std::chrono::high_resolution_clock::now();

    int elapsed_seconds =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << "Время выполнения: " << elapsed_seconds << "microseconds\n";

    cout << "MAX = " << max << "\n";
    for (int k = 0; k < arraySize; k++) {
      cpu_bin[k] = cpu_str % 2;
      cout << cpu_bin[k] << " ";
      cpu_str >>= 1;
    }
  }
  return 0;
}
