#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/device_vector.h>
#include <chrono>




extern	__global__ void run(const float* dryBuffer, const int dryBufferSize,const  float* irBuffer, const int irBufferSize, float* d_wetBuffer);

extern	void gpuRev(const float* dryBuffer, const  float* irBuffer, const int irBufferSize, int blocks, int threads, float* out);


