#include "ConvKernel.cuh"



__global__ void d2Convolution(float* C, const float* A, const float* B, const int bufferSize, const int impulseSize) {
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

cudaError_t gpuRev2(ConvData& data) {
       d2Convolution << < data.d_blocks, data.d_threads, 0, data.stream >> > (data.d_outBuffer, data.d_dry, data.d_wet, data.d_drySize, data.d_wetSize);
   


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