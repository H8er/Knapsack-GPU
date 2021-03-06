
#include "cuda_runtime.h"
#include <iostream>
#include <chrono>
#include <fstream>

#define arraySize 31 //35 max
#define def_div 10  // 5<=X<=15
#define W 100
//#define threads_per_block 32
//#define max_blocks 32

using namespace std;

__constant__ float coefs[arraySize*2];
__global__ void hybrid(float *sh_sum_dev, long int *str_num_dev, float num_of_blocks, int* bdevX,int* global_mem_bin,int threads_per_block)
{
  float th_w_sum = 0;
   float th_v_sum = 0;
   int th_bin[arraySize];
   int best_bin[arraySize];
   extern __shared__ float sh_array[];
   float* sh_maxs = (float*)sh_array;
   int* indices = (int*)&sh_maxs[threads_per_block];
  int reached = 0;
  indices[threadIdx.x] = blockIdx.x * blockDim.x + threadIdx.x;
  __syncthreads();

long signed int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
//num_to_bin += max_blocks * n_of_it;
#pragma unroll
  for (uint i = 0; i < def_div; i++)
    {
      th_bin[i] = ((num_to_bin) >> i) % 2;
      th_w_sum += th_bin[i] * coefs[i];
      th_v_sum += th_bin[i] * coefs[i+arraySize];
      best_bin[i] = th_bin[i];
    }
#pragma unroll
    for (uint i = def_div; i < arraySize; i++)
      {
        th_bin[i] = -1;
      }
int Capacity = W - th_w_sum;
sh_maxs[threadIdx.x] = (th_w_sum > W) ? 0:th_v_sum;
__syncthreads ();

//H_S
int h = def_div;
long int ns = 0;
bool forward;

while(h-def_div!=-1){
  ns++;
  forward = true;
  if(th_bin[h]==-1){
     th_bin[h]=1;
  }else{
  if(th_bin[h]==1){
     th_bin[h]=0;
  }else{
  if(th_bin[h]==0){
     th_bin[h]=-1;
    h--;
    forward=false;
  }
}
}
  if(h==arraySize-1){
    int cw = 0;
    int cp = 0;
    #pragma unroll
    for(int i = def_div;i<arraySize;i++){
      cp += coefs[i+arraySize] * th_bin[i];
      cw += coefs[i] * th_bin[i];
    }
    if((cw <= Capacity) &&(cp > reached)){
      reached = cp;
      #pragma unroll
      for(int i = def_div; i < arraySize; i++){
        best_bin[i] = th_bin[i];
      }
    }
  }
  else{
    int cw = 0;
    for(int i = def_div ; i < arraySize; i++){
      cw += coefs[i] * th_bin[i];
    }
    if (cw > Capacity) forward = false;
    cw = 0;
    float cp = 0;
    int nw = 0;
    int np = 0;
    #pragma unroll
    for(int i = def_div;i < arraySize;i++){
      np = th_bin[i]!=-1? th_bin[i] * coefs[i+arraySize]:coefs[i+arraySize];
      nw = th_bin[i]!=-1? th_bin[i] * coefs[i]: coefs[i];
      if(cw+nw <= Capacity){
        cw += nw;
        cp += np;
      }
      else{
        cp+=np*(Capacity-cw)/nw;
        break;
      }
    }
    int b = cp;
    if (b <= reached){
      forward = false;
    }
  }
  if(forward){if(h<arraySize-1){h++;}
              }
  }

sh_maxs[threadIdx.x] += reached;

__syncthreads();
//reduction on block
  for (uint offset = blockDim.x >> 1; offset >= 1; offset >>= 1)
    {
      if (threadIdx.x < offset)
	{
	  if (sh_maxs[threadIdx.x] < sh_maxs[threadIdx.x + offset])
	    {
	      sh_maxs[threadIdx.x] = sh_maxs[threadIdx.x + offset];
	      indices[threadIdx.x] = indices[threadIdx.x + offset];
	    }
	}
      __syncthreads ();
    }
  // write result for this block to global mem
  if(threadIdx.x == 0){
  sh_sum_dev[blockIdx.x] = sh_maxs[0];
  str_num_dev[blockIdx.x] = indices[0];
  }
  if(blockIdx.x*blockDim.x+threadIdx.x == indices[0]){
    #pragma unroll
    for(int i = 0; i < arraySize;i++){
      global_mem_bin[blockIdx.x*arraySize + i] = best_bin[i];
  }
  }
  __syncthreads();
}

__global__ void
hybrid_reduction (float *s, long int *str_num_dev,int* global_mem_bin,int threads_per_block)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ int sh_hy_data[];
  sh_hy_data[threadIdx.x] = s[ID];
  sh_hy_data[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads ();
  // do reduction in shared mem
  for (uint s = blockDim.x >>1; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
	{
	  if (sh_hy_data[threadIdx.x] < sh_hy_data[threadIdx.x + s])
	    {
	      sh_hy_data[threadIdx.x] = sh_hy_data[threadIdx.x + s];
	      sh_hy_data[threadIdx.x + threads_per_block] =
		sh_hy_data[threadIdx.x + threads_per_block + s];
	    }
	}
      __syncthreads ();
    }
  // write result for this block to global mem
  if (threadIdx.x == 0)
    {
			//if(sh_hy_data[0]>s[0]){//}&&(blockIdx.x>0)){
      s[blockIdx.x] = sh_hy_data[0];
      str_num_dev[blockIdx.x] = sh_hy_data[threads_per_block];

            #pragma unroll
            for(int i = 0; i < arraySize;i++){
             global_mem_bin[i] = global_mem_bin[(sh_hy_data[threads_per_block]/arraySize)*arraySize + i];

          }
		}

}


__global__ void
which_string (long int a, int *view_dev)
{
  view_dev[threadIdx.x] = (a>>threadIdx.x)%2;
}


void quickSortR(float* a,float* b, long N) {
// На входе - массив a[], a[N] - его последний элемент.

    long i = 0, j = N;      // поставить указатели на исходные места
    float temp, p;

    p = a[ N>>1 ];      // центральный элемент

    // процедура разделения
    do {
        while ( a[i] > p ) i++;
        while ( a[j] < p ) j--;

        if (i <= j) {
            temp = a[i]; a[i] = a[j]; a[j] = temp;
            temp = b[i]; b[i] = b[j]; b[j] = temp;
            temp = b[i+arraySize]; b[i+arraySize] = b[j+arraySize]; b[j+arraySize] = temp;
            i++; j--;
        }
    } while ( i<=j );

    // рекурсивные вызовы, если есть, что сортировать
    if ( j > 0 ) quickSortR(a,b, j);
    if ( N > i ) quickSortR(a+i,b+i, N-i);
}


    int main(){
      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, 0);
      int threads_per_block = deviceProp.warpSize;
      int max_blocks = pow(2,def_div)/threads_per_block;
      long int strSize_b = pow (2, arraySize);
      int num_of_blocks = strSize_b / threads_per_block;
      float *Sum = new float[32];	// = { 0 };
      float *sh_sum_dev;
      //float weight[31] ={ 5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107,115 };
      //float values[31] ={ 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313,321 };
      float dev_coefs[62] = {2,1,8,2,17,22,21,33,54,53,29,34,91,24,82,91,51,9,64,14,44,30,23,98,38,55,98,64,57,80,66,49,24,89,15,87,86,77,81,89,82,44,38,86,22,75,72,40,7,47,9,28,17,10,42,15,20,32,15,6,4,1};

      //float dev_coefs[60] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101,107, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305,313 };
      //float dev_coefs[58] = {5, 10, 17, 19, 20, 23, 26, 30, 32, 38, 40, 44, 47, 50, 55, 56, 56, 60, 62, 66, 70, 75, 77, 80, 81, 90,93,96,101, 10, 13, 16, 22, 30, 25, 55, 90, 110, 115, 130, 120, 150, 170, 194, 199, 194, 199, 217, 230, 248, 250, 264, 271, 279, 286,293,299,305 };

      //float *values_dev;
      long int *str_num_dev;
      long int *str_num = new long int[1];

      cout<<"sing param = "<<max_blocks<<" _ "<< threads_per_block<<"\n";
      cout<<"red param "<<1<<"  ,  "<<max_blocks<<"\n";

      float* additional_array = new float[arraySize];
      for(int i = 0; i < arraySize;i++){
      additional_array[i] = dev_coefs[i+arraySize]/dev_coefs[i];
      }

      quickSortR(additional_array,dev_coefs,arraySize-1);

	float t1,t2;
	float acceleration = 0;

      //for(int i = 0;i<arraySize*2;i++){dev_coefs[i] = 2;}

      std::chrono::time_point<std::chrono:: high_resolution_clock> start, end;
          start = std::chrono::high_resolution_clock::now();

      int* bdevX;
      cudaMalloc ((void **) &bdevX, arraySize * sizeof (int));
      int* global_mem_bin;
      cudaMalloc ((void **) &global_mem_bin, max_blocks*arraySize * sizeof (int));





      cudaMalloc ((void **) &sh_sum_dev,  num_of_blocks * sizeof (float));
      cudaMalloc ((void **) &str_num_dev, num_of_blocks * sizeof (float));
      cudaMemcpyToSymbol (coefs, dev_coefs, 2*arraySize * sizeof (float));




       hybrid <<< max_blocks, threads_per_block ,threads_per_block*2*sizeof(int)>>> (sh_sum_dev, str_num_dev, num_of_blocks,bdevX,global_mem_bin,threads_per_block);



hybrid_reduction<<<1,max_blocks,threads_per_block*2*sizeof(int)>>>(sh_sum_dev,str_num_dev,global_mem_bin,threads_per_block);
int* suda = new int[arraySize];
      cudaMemcpy (Sum, sh_sum_dev, sizeof (int), cudaMemcpyDeviceToHost);
      cudaMemcpy (str_num, str_num_dev, sizeof (long int), cudaMemcpyDeviceToHost);
      cudaMemcpy (suda, global_mem_bin, arraySize*sizeof (int), cudaMemcpyDeviceToHost);


      end = std::chrono:: high_resolution_clock::now();

          int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                                   (end-start).count();
          std::time_t end_time = std::chrono::system_clock::to_time_t(end);

          std::cout<< "Время выполнения: " << elapsed_seconds << "microseconds\n";
	t1 = elapsed_seconds;
      cout << "Acheived maximal sum = " << Sum[0] << "\n";

        for (int i = 0; i < arraySize; i++)
          {
            cout << suda[i];
          } cout << "\n";


        //check
        int checksum = 0;
        for (int i = 0; i < arraySize; i++)
          {
            checksum += dev_coefs[i+arraySize] * suda[i];
          }
        cout << "Validation sum = " << checksum << "\n";
        checksum = 0;
        for (int i = 0; i < arraySize; i++)
          {
            checksum += dev_coefs[i] * suda[i];
          } cout << "Weight = " << checksum << "\n";
         // ofstream fout;
         // fout.open("data_uncorr_hybrid.txt",ios_base::app);

         // fout<<"GPU\n"<<Sum[0]<<"\n"<<elapsed_seconds<<"\n\n";



        cudaFree(coefs);
        cudaFree (sh_sum_dev);
        cudaFree (str_num_dev);
        cudaFree(bdevX);
        cudaFree(global_mem_bin);

        delete [] Sum;
        delete [] str_num;


        cout<<"Проверка. CPU version:\n";
        start = std::chrono::high_resolution_clock::now();
        int *X = new int[arraySize];
        int *bestX = new int[arraySize];
        for(int i = 0; i < arraySize; i++){
          X[i] = -1;
          bestX[i] = 0;
        }
        int curr_sum = 0;
        int reached_max = 0;
        float *cpu_bin = new float[arraySize];

        for(int i = 0; i < arraySize;i++){
        additional_array[i] = dev_coefs[i+arraySize]/dev_coefs[i];
        }
        quickSortR(additional_array,dev_coefs,arraySize-1);

        int h = 0;
        int k = h;//def_div;
        long int ns = 0;
        bool forward;
        while(h-k!=-1){
          ns++;
          forward = true;
          if(X[h]==-1){
            X[h]=1;
          }else{
          if(X[h]==1){
            X[h]=0;
          }else{
          if(X[h]==0){
            X[h]=-1;
            h--;
            forward=false;
          }
        }
        }
          if(h==arraySize-1){
            int cw = 0;
            int cp = 0;
            for(int i = k;i<arraySize;i++){
              cp += dev_coefs[i+arraySize]*X[i];
              cw += dev_coefs[i]*X[i];
            }
            if((cw <= W) &&(cp > reached_max)){
              reached_max = cp;
              for(int i = k; i < arraySize; i++){
                bestX[i] = X[i];
              }
            }
          }
          else{
            int cw = 0;
            for(int i = k ; i < arraySize; i++){
              cw += dev_coefs[i]*X[i];
            }
            if (cw > W) forward = false;
            cw = 0;
            float cp = 0;
            int nw = 0;
            int np = 0;
            for(int i = k;i<arraySize;i++){
              np = X[i]!=-1? X[i] * dev_coefs[i+arraySize]:dev_coefs[i+arraySize];
              nw = X[i]!=-1? X[i] * dev_coefs[i]: dev_coefs[i];
              if(cw+nw <= W){
                cw += nw;
                cp += np;
              }
              else{
                cp+=np*(W-cw)/nw;
                break;
              }
            }
            int b = cp;
            if (b <= reached_max){
              forward = false;
            }
          }
          if(forward){if(h<arraySize-1){h++;}}
          }


          end = std::chrono:: high_resolution_clock::now();

              elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                                       (end-start).count();
               end_time = std::chrono::system_clock::to_time_t(end);
		t2 = elapsed_seconds;
          std::cout<< "Время выполнения: " << elapsed_seconds << "microseconds\n";

        cout<<"MAX = "<<reached_max<<"\n";
        for(int m = 0 ; m < arraySize;m++){
        cout<<bestX[m];
        curr_sum += bestX[m]*dev_coefs[m+arraySize];
        }cout<<"\nЧисло итераций = "<<ns<<"\n";


       // fout<<"CPU\n"<<reached_max<<"\n"<<elapsed_seconds<<"\n\n";
      //  fout.close();
acceleration = t2/t1;
cout<<"Acceleration = "<<acceleration<<"\n";
delete [] suda;
delete [] additional_array;

cudaFree (sh_sum_dev);
cudaFree (str_num_dev);
cudaFree (coefs);

return 0;
}
