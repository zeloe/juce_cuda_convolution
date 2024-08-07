#include <stdlib.h> 

#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <stdio.h>



#ifndef _GPUConvEngine_H_

#define _GPUConvEngine_H_
 

__global__ void shared_partitioned_convolution1(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);
__global__ void shared_partitioned_convolution2(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);
__global__ void shared_partitioned_convolution3(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);
__global__ void shared_partitioned_convolution4(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp);
__global__ void  shiftAndInsertKernel(float* __restrict__ delayBuffer);

class GPUConvEngine {
public:
	GPUConvEngine();
	~GPUConvEngine();

	GPUConvEngine(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize);
	
	void  process(float* in);
	 
	float* h_result_ptr = nullptr;
private:
	void   launchEngine();
	void checkCudaError(cudaError_t err, const char* errMsg);
	
	 
	int bs = 0;
	int h_numPartitions = 0;
	int h_paddedSize = 0;
	int h_convResSize = 0;
	int h_SizeOfSubPartitions = 0;
 
	const int numOfSubPartitions = 4;
	cudaStream_t streams[4] = { nullptr }; // Initialize to nullptr
	int* h_sizesOfSubPartitions = nullptr;
	
	
	float tempScale;
	float* d_IR_padded = nullptr;
	float* d_TimeDomain_padded = nullptr;
	size_t SHMEM = 0;
	float* d_ConvolutionRes = nullptr;
	float* h_ConvolutionRes = nullptr;
	float* h_Overlap = nullptr;
	float* d_Input = nullptr;
 
	dim3 dThreads;
	dim3 dBlocks;
	dim3 threadsPerBlock;
	dim3 numBlocks;
	
};



#endif