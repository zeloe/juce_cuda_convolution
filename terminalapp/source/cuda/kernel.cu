#include "kernel.h"


 __global__ void linearConvKernel( float* x1,  float* x2, const int* size, float* result)
{
   	const int c = blockIdx.x * blockDim.x + threadIdx.x; // samples

    	if (c < *size) {
       		 
        	float val = 0;

        	for (int i = 0; i < *size; i++) {
            		int index = (c - i + *size) % *size;
            		val += x1[i] * x2[index];
        }

        	result[c] = val * 0.015;
    }
   
} 





 __global__ void cu_mult(float* x1, float* scale, const  int* channels, const  int* size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    for(unsigned int ch = 0; ch < *channels; ch++)
    {
        int index = ch * (*size) + id;
        x1[index] *= *scale;
    }
}


 __global__ void copy(float* big, float* small, int* size)
 {
     int id = blockIdx.x * blockDim.x + threadIdx.x;

 
     if (id < *size) {
      
         big[id] = small[id];
     }
 }


 void gpuRev(float* dryBuffer, float* irBuffer, const int bufferSize, float* out, int* d_size) {

     int THREADS = 512;
 
     int GRID = (bufferSize + THREADS - 1) / THREADS;
 
     // Perform circular convolution
     linearConvKernel << <GRID, THREADS>> > (dryBuffer, irBuffer, d_size, out);
     cudaDeviceSynchronize();
 }



 __global__ void partConvKernel(float* x1, float* x2, const int* size, float* result, int* numPartitions)
 {
     const int c = blockIdx.x * blockDim.x + threadIdx.x; // samples
      
     int counter = 0;
     if (c < *size * (*numPartitions)) {
         int offset = *size * counter;
         float val = 0;

         for (int i = 0; i < *size; i++) {
             int index = (c - i + *size + offset) % (*size * (*numPartitions));
             val += x1[i] * x2[index];
         }
         
         result[c + offset] += val * 0.015;
         counter++;
     }
     
 }





 void partgpuRev(float* dryBuffer, float* partirBuffer, const int bufferSize, float* out, int* d_size, int* d_numPartitions, int h_numPartitions) {

     int THREADS = 1024;
 
     int GRID = ((bufferSize * h_numPartitions) + THREADS - 1) / THREADS;


     partConvKernel <<< GRID, THREADS >>> (dryBuffer, partirBuffer, d_size, out, d_numPartitions);
     cudaDeviceSynchronize();


 }
