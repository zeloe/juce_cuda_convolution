#include "ConvKernel.cuh"



__global__ void d2Convolution(float* C, const float* A, const float* B, const int bufferSize, const int impulseSize, float normFactor) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
    int tidy = threadIdx.y + blockIdx.y * blockDim.y; // Global thread index

    int sliceSize = 2 * bufferSize - 1;
    int totalSlices = impulseSize / bufferSize;

    if (tidx < totalSlices && tidy < sliceSize) {
        int slice = tidx;
        int index_within_slice = tidy;
        float sum = 0.0f;

        int start_idx = max(0, index_within_slice - bufferSize + 1);
        int end_idx = min(index_within_slice + 1, bufferSize);

        for (int j = start_idx; j < end_idx; j++) {
            int k = index_within_slice - j;
            sum += (A[j + slice * bufferSize] * B[k + slice * bufferSize]);
        }

        // Use atomicAdd to accumulate results correctly across threads
        atomicAdd(&C[index_within_slice], sum);
    }
}







__global__ void Convolution(float* C, const float* A, const float* B, const int smallSize, const int bigSize) {


    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index

    int sliceSize = 2 * smallSize - 1;
    int totalSlices = bigSize / smallSize;

    if (tid < totalSlices * sliceSize) {
        int slice = tid / sliceSize;
        int index_within_slice = tid % sliceSize;
        float sum = 0.0f;

        int start_idx = max(0, index_within_slice - smallSize + 1);
        int end_idx = min(index_within_slice + 1, smallSize);

        for (int j = start_idx; j < end_idx; j++) {
            int k = index_within_slice - j;
            sum += (A[j + slice * smallSize] * B[k + slice * smallSize]);
        }
        __syncthreads();
        // Write the convolution result within the valid range of the slice
        C[tid] = sum;
    }
}


__global__ void convKernel(const float* dryBuffer, const int dryBufferSize, const float* irBuffer, const int irBufferSize, float* d_wetBuffer)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    if (id < irBufferSize) {

        int start = id < irBufferSize ? 0 : id - irBufferSize;
        double val = 0;

        for (int i = start; i < id; i++) {
            if (i >= dryBufferSize) break;

            int irBufferPosition = (irBufferSize - 1) + (start - i);

            val += dryBuffer[i] * irBuffer[irBufferPosition];
        }

        d_wetBuffer[id] = val;
    }
}

__global__ void InputSideConvolution(float* C, const float* B, const float* A, const int smallSize, const int totalSize) {


}



cudaError_t gpuRev(float* dryBuffer, float* irBuffer, const int totalSize, float* out, int irSize, int trackSize) {
    cudaError_t cudaStatus;
    return cudaStatus;
}



cudaError_t gpuRev2(ConvData& data) {
       size_t SHMEM = data.d_drySize * sizeof(float) * 2;
       d2Convolution << < data.d_blocks, data.d_threads, 0, data.stream >> > (data.d_outBuffer, data.d_dry, data.d_wet, data.d_drySize, data.d_wetSize,data.normFactor);
    //   convolveOVS << <data.d_blocks, data.d_threads, SHMEM, data.stream >> > (data.d_dry, data.d_result, data.d_wet, 0, data.d_drySize, data.d_wetSize, data.d_drySize);
    //   Convolution <<< data.d_blocks.x, 1024, 0, data.stream >>> (data.d_result, data.d_wet, data.d_dry, data.d_drySize, data.d_wetSize);
      //  dConvolution << <data.d_blocks, data.d_threads,0,data.stream >> > (data.d_result, data.d_dry, data.d_wet, data.d_drySize, data.d_wetSize);
     // reduceKernel << < data.reduceBlocks, data.reduceThreads,0, data.stream >> > (data.d_result, data.d_outBuffer, data.d_drySize, data.d_wetSize);


#ifdef _DEBUG
    // Check for any errors encountered during the launch.
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Convolution kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // Synchronize to ensure all kernels have finished
    cudaStatus = cudaStreamSynchronize(data.stream);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Convolution kernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaSuccess;

#endif

}