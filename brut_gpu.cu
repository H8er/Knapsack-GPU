#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <sstream>



#define arraySize 31 //35 max
//#define W 1741

using namespace std;

__constant__ float coefs[arraySize*2+1];
__global__ void single_thread(float *sh_sum_dev,long int *str_num_dev, float num_of_blocks, int rep,int threads_per_block,int max_blocks)
{
  float th_w_sum = 0;
   float th_v_sum = 0;
   float th_bin[arraySize];
   int n_of_it = rep;
  extern __shared__ float sh_array[];
  float* sh_maxs = (float*)sh_array;
  long int* indices = (long int*)&sh_maxs[threads_per_block];
  indices[threadIdx.x] = threadIdx.x;


long signed int num_to_bin = blockIdx.x * blockDim.x + threadIdx.x;
num_to_bin += max_blocks * n_of_it;
__syncthreads();
#pragma unroll
  for (uint i = 0; i < arraySize; i++)
    {
      th_bin[i] = ((num_to_bin) >> i) % 2;
      th_w_sum += th_bin[i] * coefs[i];
      th_v_sum += th_bin[i] * coefs[i+arraySize];
    }

sh_maxs[threadIdx.x] = (th_w_sum > coefs[arraySize*2]) ? 0:th_v_sum;

__syncthreads ();

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
  sh_sum_dev[blockIdx.x+max_blocks*rep] = sh_maxs[0];
  //str_num_dev[blockIdx.x+max_blocks*rep] = indices[0]+max_blocks*rep;
}
if(threadIdx.x == indices[0]){str_num_dev[blockIdx.x+max_blocks*rep] = num_to_bin;}

}

__global__ void
reduction_max (float *s, long int *str_num_dev,int threads_per_block)
{
  int ID = blockIdx.x * blockDim.x + threadIdx.x;
  extern __shared__ float sdata[];
  sdata[threadIdx.x] = s[ID];
  sdata[threadIdx.x + threads_per_block] = str_num_dev[ID];

  __syncthreads ();
  // do reduction in shared mem
  for (uint s = blockDim.x >>1; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
	{
	  if (sdata[threadIdx.x] < sdata[threadIdx.x + s])
	    {
	      sdata[threadIdx.x] = sdata[threadIdx.x + s];
	      sdata[threadIdx.x + threads_per_block] =
		sdata[threadIdx.x + threads_per_block + s];
	    }
	}
      __syncthreads ();
    }
  // write result for this block to global mem
  if (threadIdx.x == 0)
    {
			//if(sdata[0]>s[0]){//}&&(blockIdx.x>0)){
      s[blockIdx.x] = sdata[0];
      str_num_dev[blockIdx.x] = sdata[threads_per_block];
		}

    //}
}

__global__ void
which_string (int a, int *view_dev)
{
  view_dev[threadIdx.x] = (a >> threadIdx.x) % 2;
}


int main(){
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int threads_per_block = deviceProp.maxThreadsDim[0];
  int max_blocks = deviceProp.maxGridSize[0]/2 + 1;
  long int strSize_b = pow (2, arraySize);
  int num_of_blocks = strSize_b / threads_per_block;
  float *Sum = new float[1];	// = { 0 };
  float *sh_sum_dev;

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


  //float *values_dev;
  long int *str_num_dev;
  long int *str_num =  new long int[1];
  float N_of_rep;
  N_of_rep = num_of_blocks/max_blocks>0?num_of_blocks/max_blocks:1;
  int sing_blocks = num_of_blocks/N_of_rep>0?num_of_blocks/N_of_rep:1;

//for(int i = 0;i<arraySize*2;i++){dev_coefs[i] = 2;}

  std::chrono::time_point<std::chrono:: high_resolution_clock> start, end;

      start = std::chrono::high_resolution_clock::now();

  cudaMalloc ((void **) &sh_sum_dev,  num_of_blocks * sizeof (float));
  cudaMalloc ((void **) &str_num_dev, num_of_blocks * sizeof (long));
  cudaMemcpyToSymbol (coefs, dev_coefs, (2*arraySize + 1) * sizeof (float));



        for(int i = 0;i<N_of_rep;i++){
          //cout<<i;
  single_thread <<< sing_blocks, threads_per_block,threads_per_block*3*sizeof(int) >>> (sh_sum_dev, str_num_dev, num_of_blocks,i,threads_per_block,max_blocks);
             }

int k = num_of_blocks/threads_per_block;
while(k>=1){
//cout<<k<<" ";

               if(k>=threads_per_block){
                reduction_max <<<k, threads_per_block,threads_per_block*3*sizeof(int)>>> (sh_sum_dev, str_num_dev,threads_per_block);
                 k/=threads_per_block;}
               else break;
             }
if(k>1){
reduction_max <<<1,k,k*2*sizeof(int)>>> (sh_sum_dev, str_num_dev,k);
}

  cudaMemcpy (Sum, sh_sum_dev, sizeof (float), cudaMemcpyDeviceToHost);
  cudaMemcpy (str_num, str_num_dev, sizeof (float), cudaMemcpyDeviceToHost);

  end = std::chrono:: high_resolution_clock::now();

      int elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>
                               (end-start).count();
      std::time_t end_time = std::chrono::system_clock::to_time_t(end);

      std::cout<< "Время выполнения: " << elapsed_seconds << "microseconds\n";

  cout << "Acheived maximal sum = " << Sum[0] << "\n";
  cout << "String number " << int(str_num[0]) << "\n";

  int *view = new int[arraySize];
  int *view_dev;
  cudaMalloc ((void **) &view_dev, arraySize * sizeof (int));
  which_string <<< 1, arraySize >>> (str_num[0], view_dev);
  cudaMemcpy (view, view_dev, arraySize * sizeof (int),
	      cudaMemcpyDeviceToHost);
  for (int i = 0; i < arraySize; i++)
    {
      cout << view[i] << " ";
    } cout << "\n";
  //check
  float checksum = 0;
  for (int i = 0; i < arraySize; i++)
    {
      checksum += dev_coefs[i+arraySize] * view[i];
    }
  cout << "Validation sum = " << checksum << "\n";
  checksum = 0;
  for (int i = 0; i < arraySize; i++)
    {
      checksum += dev_coefs[i] * view[i];
    } cout << "Weight = " << checksum << "\n";

  cudaFree (sh_sum_dev);
  cudaFree (str_num_dev);
  cudaFree (coefs);
  cudaFree (view_dev);
}
  return 0;
}
