
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
	dim3 reduceThreads;
	dim3 reduceBlocks;
	cudaStream_t stream;
	float normFactor;
};




__global__ void d2Convolution(float* C, const float* A, const float* B, const int bufferSize, const int impulseSize, float normFactor);
 cudaError_t gpuRev(float* dryBuffer, float* irBuffer, const int totalSize, float* out, int irSize, int trackSize);
 cudaError_t gpuRev2(ConvData& data);
 __global__ void Convolution (float* C, const float* A, const float* B, const int smallSize, const int bigSize);
 __global__ void InputSideConvolution(float* C, const float* B, const float* A, const int smallSize, const int totalSize);
 __global__ void convKernel(const float* dryBuffer, const int dryBufferSize, const float* irBuffer, const int irBufferSize, float* d_wetBuffer);
#endif