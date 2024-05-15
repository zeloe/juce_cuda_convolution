#ifndef _KERNEL_H_

#define _KERNEL_H_
#include "cuda_runtime.h"

#include "device_launch_parameters.h"
#include <iostream>
#include <chrono>




 extern "C" __global__ void linearConvKernel(float* x1, float* x2, const int* size, float* result);

 extern "C" __global__ void cu_mult(float* x1, float* scale, const  int* channels, const  int* size);

 extern "C"   void gpuRev(float* dryBuffer, float* irBuffer, const int bufferSize, float* out, int* d_size);


 extern "C" __global__ void copy(float* big, float* small, int* size);

 extern "C" void partgpuRev(float* dryBuffer, float* partirBuffer, const int bufferSize, float* out, int* d_size, int* d_numPartitions, int h_numPartitions);
 extern "C" __global__ void partConvKernel(float* x1, float* x2, const int* size, float* result, int* d_numPartitions);


#endif