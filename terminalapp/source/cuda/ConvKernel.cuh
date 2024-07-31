
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

#include <stdio.h>


#ifndef _CONVKERNEL_H_

#define _CONVKERNEL_H_
// device dry pointer,device wet pointer, device result pointer, device dry size, device wetSize, d_threads, d_blocks, stream
struct ConvData{
	float* d_dry;
	float* d_wet;
	float* d_result;
	int d_drySize;
	int d_wetSize;
	dim3 d_threads;
	dim3 d_blocks;
	float* d_outBuffer;
	cudaStream_t stream;
};




__global__ void d2Convolution(float* C, const float* A, const float* B, const int bufferSize, const int impulseSize);

 cudaError_t gpuRev2(ConvData& data);
#endif