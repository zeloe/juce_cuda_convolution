
#include "GPUConvEngine.cuh"
// Define the constant memory array
__constant__ int SIZES[2];
__constant__ float INPUT[1024];
__global__ void shared_partitioned_convolution(float* __restrict__ Result, const float* __restrict__ Dry, const float* __restrict__ Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];

	// Declare pointers to the shared memory partitions
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	float* tempResult = &partArray[SIZES[0] * 2];
	// Load data into shared memory
	tempResult[copy_idx] = 0.f;
	tempResult[SIZES[0] + copy_idx] = 0.f;
	arr1[copy_idx] = Dry[thread_idx];
	arr2[copy_idx] = Imp[thread_idx];

	__syncthreads();

	// Shared memory to accumulate results before writing them to global memory
	// Convolution operation (reduction into shared memory)
	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (i + copy_idx) % SIZES[0];
		tempResult[i + inv] += arr1[i] * arr2[inv];
	}

	__syncthreads();  // Ensure all threads in the block have finished processing


	// Write the accumulated result to global memory (only for the first thread)
	if (copy_idx == 0) {
		// Write the first part of the result (up to SIZES[0] - 1)
		for (int i = 0; i < SIZES[0] - 1; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}

		// Write the second part of the result (for the overlap, from SIZES[0] - 1 to 2 * SIZES[0] - 2)
		for (int i = SIZES[0] - 1; i < 2 * SIZES[0] - 1; i++) {
			atomicAdd(&Result[i], tempResult[i]);
		}
	}
}

__global__ void  shiftAndInsertKernel(float* __restrict__ delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES[0]) {
		delayBuffer[tid] = INPUT[tid];
	}
	
		delayBuffer[tid + SIZES[0]] = delayBuffer[tid];

	
}

GPUConvEngine::GPUConvEngine() {

}



GPUConvEngine::~GPUConvEngine() {
	 
	// Free Stream 
	cudaStreamDestroy(stream);
	cudaFree(d_IR_padded);
	cudaFree(SIZES);
	cudaFree(d_TimeDomain_padded);
	cudaFree(d_ConvolutionRes);
	cudaFree(d_Input);

	free(h_ConvolutionRes);
	free(h_Overlap);
	free(h_sizesOfSubPartitions);
}

void GPUConvEngine::checkCudaError(cudaError_t err, const char* errMsg) {
	if (err != cudaSuccess) {
		printf("CUDA Error (%s): %s\n", errMsg, cudaGetErrorString(err));
	}
}
GPUConvEngine::GPUConvEngine(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize) {
	cudaStreamCreate(&stream);
	h_convResSize = maxBufferSize * 2 - 1;
	floatSizeRes = h_convResSize * sizeof(float);
	checkCudaError(cudaMalloc((void**)&d_ConvolutionRes, floatSizeRes), "d_ConvolutionRes malloc");
	checkCudaError(cudaMemset(d_ConvolutionRes, 0, floatSizeRes), "d_ConvolutionRes memset");
	SHMEM = 2 * sizeof(float) * maxBufferSize + floatSizeRes;
	bs = maxBufferSize;
	bs_float = bs * sizeof(float);
	int* cpu_sizes = (int*)calloc(2, sizeof(int));
	h_result_ptr = (float*)calloc(bs, sizeof(float));
	h_ConvolutionRes = (float*)calloc(h_convResSize, sizeof(float));
	h_Overlap = (float*)calloc(bs, sizeof(float));

	cpu_sizes[0] = bs;
	checkCudaError(cudaMalloc((void**)&d_Input, bs * sizeof(float)), "d_Input malloc");
	checkCudaError(cudaMemset(d_Input, 0, bs * sizeof(float)), "d_Input memset");

	h_numPartitions = (int((impulseResponseSize) / bs) + 1);
	h_paddedSize = h_numPartitions * bs;

	h_SizeOfSubPartitions = h_paddedSize / numOfSubPartitions;
	 

	 
	cpu_sizes[1] = h_paddedSize;
	cudaMemcpyToSymbol(SIZES, cpu_sizes, 2 *  sizeof(int));
	 
	printf("Number of operations for each audioblock ~ %d\n", h_paddedSize * h_paddedSize);

	checkCudaError(cudaMalloc((void**)&d_IR_padded, h_paddedSize * sizeof(float)), "d_IR_padded malloc");
	checkCudaError(cudaMemset(d_IR_padded, 0, h_paddedSize * sizeof(float)), "d_IR_padded memset");
	checkCudaError(cudaMemcpy(d_IR_padded, impulseResponseBufferData, impulseResponseSize * sizeof(float), cudaMemcpyHostToDevice), "d_IR_padded memcpy");

	checkCudaError(cudaMalloc((void**)&d_TimeDomain_padded, h_paddedSize * sizeof(float)), "d_TimeDomain_padded malloc");
	checkCudaError(cudaMemset(d_TimeDomain_padded, 0, h_paddedSize * sizeof(float)), "d_TimeDomain_padded memset");

	for (int i = 0; i < numOfSubPartitions; i++) {
		checkCudaError(cudaStreamCreate(&streams[i]), "stream create");
	}
	dThreads.x = bs;
 
	dBlocks.x = (h_numPartitions);
 
	threadsPerBlock.x = 256;
	numBlocks.x = (h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
	free(cpu_sizes);
}




void  GPUConvEngine::process(float* in, float* const* outputChannelData) {

	//copy content and transfer
	cudaMemcpyToSymbolAsync(INPUT, in, bs_float, 0, cudaMemcpyHostToDevice, stream);




	//launch the convolution Engine
	launchEngine();
	float* outA = outputChannelData[0];
	float* outB = outputChannelData[1];
	__m128 scale = _mm_set1_ps(0.015f); // Load scaling factor into an SSE register

	for (int i = 0; i < bs; i += 4) {
		// Load 4 floats from h_ConvolutionResL and h_OverlapL
		__m128 res = _mm_loadu_ps(&h_ConvolutionRes[i]);
		__m128 overlap = _mm_loadu_ps(&h_Overlap[i]);

		// Perform (resL + overlapL) * scale for left channel
		__m128 result = _mm_mul_ps(_mm_add_ps(res, overlap), scale);
		_mm_storeu_ps(&outA[i], result); // Store the result in outA
		_mm_storeu_ps(&outB[i], result); // Store the result in outB
		 

	}

	// Copy the last `bs` elements as overlap values for the next block
	std::memcpy(h_Overlap, &h_ConvolutionRes[bs - 1], bs_float);


}




void  GPUConvEngine::launchEngine() {
	shiftAndInsertKernel << <numBlocks, threadsPerBlock, 0, stream >> > (d_TimeDomain_padded);
	shared_partitioned_convolution << <dBlocks, dThreads, SHMEM, stream >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
	checkCudaError(cudaGetLastError(), "partitionedSubConvolution launch");
	 
	 
		 
 
	// Copy the convolution results back to the host
			cudaMemcpyAsync(h_ConvolutionRes, d_ConvolutionRes, floatSizeRes, cudaMemcpyDeviceToHost, stream);
	//set the result to 0

			cudaMemsetAsync(d_ConvolutionRes, 0, floatSizeRes, stream);

			cudaStreamSynchronize(stream);
}
