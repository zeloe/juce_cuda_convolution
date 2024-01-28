#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>




extern	__global__ void circularConvKernel( float* x1,  float* x2, const int* size, float* result, const  int * channels);

extern	__global__ void cu_mult(float* x1, float* scale, const  int* channels, const  int* size);

extern	void gpuRev(float* dryBuffer,float* irBuffer, const int bufferSize, float* out, const unsigned int channels);


