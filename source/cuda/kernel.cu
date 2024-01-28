
#include "kernel.h"



__global__ void circularConvKernel( float* x1,  float* x2, const int* size, float* result, const  int* channels)
{
   	const int c = blockIdx.x * blockDim.x + threadIdx.x; // samples
    	int ch = 0;

    	if (c < *size * *channels) {
       		int offset = ch  * (*size); // Offset to the start of the current channel

        	float val = 0;

        	for (int i = 0; i < *size; i++) {
            		int index = (c - i + *size) % *size;
            		val += x1[offset + i] * x2[offset + index];
        }

        	result[offset + c] = val;
        	 ch++;
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
void gpuRev(float* dryBuffer,float* irBuffer, const int bufferSize, float* out, const unsigned int channels)
{

	int threads = 1024;

	int blocks = ((int)(bufferSize * channels) / threads) + 1;


    	float* d_wetBuffer;
    	float* d_dryBufferC;
    	float* d_impBufferC;
    	
    	int* d_size;
    	cudaMalloc((void**)&d_size, sizeof(int));

    	int* d_channels;
    	cudaMalloc((void**)&d_channels, sizeof( int));

    	float* d_scale;
    	float h_scale = 0.15;
    	cudaMalloc((void**)&d_scale, sizeof(float));
    	cudaMemcpy(d_scale, &h_scale, sizeof(float), cudaMemcpyHostToDevice);

    // Copy to GPU
    	cudaMemcpy(d_size, &bufferSize, sizeof(int), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_channels, &channels, sizeof( int), cudaMemcpyHostToDevice);


	cudaMalloc( (void**)&d_wetBuffer ,bufferSize * channels*sizeof(float) );
	cudaMalloc( (void**)&d_dryBufferC ,bufferSize * channels*sizeof(float) );
	cudaMalloc( (void**)&d_impBufferC , bufferSize * channels*sizeof(float) );
	
	cudaMemcpy(d_dryBufferC,dryBuffer,bufferSize * channels*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_impBufferC,irBuffer,bufferSize * channels*sizeof(float),cudaMemcpyHostToDevice);
	 

	// Perform circular convolution
	circularConvKernel<<<blocks, threads>>>(d_dryBufferC, d_impBufferC, d_size, d_wetBuffer, d_channels);
	cudaDeviceSynchronize();

	// Perform multiplication/normalisation
	//cu_mult<<<gridSize_mult, blockSize_mult>>>(d_wetBuffer, d_scale, d_channels, d_size);
	//cudaDeviceSynchronize();

    // Copy result back to host
    	cudaMemcpy(out,d_wetBuffer,bufferSize * channels*sizeof(float), cudaMemcpyDeviceToHost);

     

    cudaFree(d_size);
    cudaFree(d_channels);
    cudaFree(d_scale);
    cudaFree(d_wetBuffer);
    cudaFree(d_dryBufferC);
    cudaFree(d_impBufferC);
}
