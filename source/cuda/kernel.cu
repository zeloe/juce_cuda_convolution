
#include "kernel.h"



__global__ void circularConvKernel(const float* x1, const float* x2, const int* size, float* result)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < *size) {
        float val = 0;

        for (int i = 0; i < *size; i++) {
            int index = (id - i + *size) % *size;
            val += x1[i] * x2[index];
        }
        
        result[id] = val;
    }
}

void gpuRev(const float* dryBuffer,const  float* irBuffer, const int irBufferSize, int blocks, int threads, float* out)
 {
 
 	 


	float* d_wetBuffer;
	cudaMalloc((void**)&d_wetBuffer, (irBufferSize) * sizeof(float));

	float* d_dryBufferC;
	cudaMalloc((void**)&d_dryBufferC, (irBufferSize) * sizeof(float));
	
	float* d_impBufferC;
	cudaMalloc((void**)&d_impBufferC, (irBufferSize) * sizeof(float));
	
	int* d_size;
	cudaMalloc((void**)&d_size, sizeof(int));
	
	//copy to gpu
	cudaMemcpy(d_size, &irBufferSize, sizeof(int), cudaMemcpyHostToDevice);
 	cudaMemcpy(d_dryBufferC, dryBuffer, (irBufferSize) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_impBufferC, irBuffer, (irBufferSize) * sizeof(float), cudaMemcpyHostToDevice);
 	
 	//perform
	 circularConvKernel<<<blocks, threads>>>(d_dryBufferC,d_impBufferC, d_size, d_wetBuffer);
	 cudaDeviceSynchronize();
	 // Wait for GPU to finish before accessing on host


	 
	
	 cudaMemcpy(out, d_wetBuffer, (irBufferSize) * sizeof(float), cudaMemcpyDeviceToHost);


	  
	 
	// Free device and host memory
    	 
   	 cudaFree(d_dryBufferC);
   	 cudaFree(d_impBufferC);
   	 cudaFree(d_wetBuffer);
   	 cudaFree(d_size);
    
}


