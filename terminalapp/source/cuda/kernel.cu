#include "kernel.h"


 __global__ void circularConvKernel( float* x1,  float* x2, const int* size, float* result)
{
   	const int c = blockIdx.x * blockDim.x + threadIdx.x; // samples

    	if (c < *size) {
       		 
        	float val = 0;

        	for (int i = 0; i < *size; i++) {
            		int index = (c - i + *size) % *size;
            		val += x1[i] * x2[index];
        }

        	result[c] = val;
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

  void gpuRev(float* dryBuffer,float* irBuffer, const int bufferSize, float* out, int* d_size)
{

	int threads = 1024;

	int blocks = ((int)(bufferSize) / threads) + 1;

	// Perform circular convolution
	circularConvKernel<<<blocks, threads>>>(dryBuffer, irBuffer, d_size, out);
	cudaDeviceSynchronize();

	// Perform multiplication/normalisation
	//cu_mult<<<gridSize_mult, blockSize_mult>>>(d_wetBuffer, d_scale, d_channels, d_size);
	//cudaDeviceSynchronize();

     

    
}
