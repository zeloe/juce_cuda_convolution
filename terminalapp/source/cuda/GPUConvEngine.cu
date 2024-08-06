
#include "GPUConvEngine.cuh"
// Define the constant memory array
__constant__ int SIZES[2];
__constant__ int OFFSETS[4];
__constant__ float INPUT[1024];
__global__ void shared_partitioned_convolution1(float* Result, const float* Dry, const float* Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	arr1[copy_idx] = Dry[thread_idx +  OFFSETS[0]];
	arr2[copy_idx] = Imp[thread_idx + OFFSETS[0]];


	__syncthreads();

	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (copy_idx - i) % SIZES[0];
		atomicAdd(&Result[i + inv], arr1[i] * arr2[inv]);
	}
}

__global__ void shared_partitioned_convolution2(float* Result, const float* Dry, const float* Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	arr1[copy_idx] = Dry[thread_idx + OFFSETS[1]];
	arr2[copy_idx] = Imp[thread_idx + OFFSETS[1]];


	__syncthreads();

	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (copy_idx - i) % SIZES[0];
		atomicAdd(&Result[i + inv], arr1[i] * arr2[inv]);
	}
}

__global__ void shared_partitioned_convolution3(float* Result, const float* Dry, const float* Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	arr1[copy_idx] = Dry[thread_idx + OFFSETS[2]];
	arr2[copy_idx] = Imp[thread_idx + OFFSETS[2]];


	__syncthreads();

	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (copy_idx - i) % SIZES[0];
		atomicAdd(&Result[i + inv], arr1[i] * arr2[inv]);
	}
}

__global__ void shared_partitioned_convolution4(float* Result, const float* Dry, const float* Imp) {
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int copy_idx = threadIdx.x;
	extern __shared__ float partArray[];
	float* arr1 = &partArray[0];
	float* arr2 = &partArray[SIZES[0]];
	arr1[copy_idx] = Dry[thread_idx + OFFSETS[3]];
	arr2[copy_idx] = Imp[thread_idx + OFFSETS[3]];


	__syncthreads();

	for (int i = 0; i < SIZES[0]; i++) {
		int inv = (copy_idx - i) % SIZES[0];
		atomicAdd(&Result[i + inv], arr1[i] * arr2[inv]);
	}
} 

__global__ void  shiftAndInsertKernel(float* delayBuffer) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Shift elements within the delay buffer
	if (tid < SIZES[1] - SIZES[0]) {
		delayBuffer[tid + SIZES[0]] = delayBuffer[tid];
	}

	// Insert new elements at the beginning of the delay buffer
	if (tid < SIZES[0]) {
		delayBuffer[tid] = INPUT[tid];
	}
}

GPUConvEngine::GPUConvEngine() {

}



GPUConvEngine::~GPUConvEngine() {
	for (int i = 0; i < numOfSubPartitions; i++) {
		cudaStreamDestroy(streams[i]);
	}

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
	h_convResSize = maxBufferSize * 2 - 1;
	int floatSizeRes = h_convResSize * sizeof(float);
	checkCudaError(cudaMalloc((void**)&d_ConvolutionRes, floatSizeRes), "d_ConvolutionRes malloc");
	checkCudaError(cudaMemset(d_ConvolutionRes, 0, floatSizeRes), "d_ConvolutionRes memset");
	SHMEM = 2 * sizeof(float) * maxBufferSize;
	bs = maxBufferSize;
	
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

	int numBerOFSubPArtitions = (h_SizeOfSubPartitions / bs) + 1;
	h_paddedSize = numBerOFSubPArtitions * numOfSubPartitions * bs;
	h_SizeOfSubPartitions = h_paddedSize / numOfSubPartitions;


	h_sizesOfSubPartitions = (int*)malloc(numOfSubPartitions * sizeof(int));
	for (int i = 0; i < numOfSubPartitions; i++) {
		h_sizesOfSubPartitions[i] = h_SizeOfSubPartitions * (i);
	}
	cpu_sizes[1] = h_paddedSize;
	cudaMemcpyToSymbol(SIZES, cpu_sizes, 2 *  sizeof(int));
	cudaMemcpyToSymbol(OFFSETS, h_sizesOfSubPartitions, numOfSubPartitions * sizeof(int));
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
 
	dBlocks.x = (numBerOFSubPArtitions);
 
	threadsPerBlock.x = 256;
	numBlocks.x = (h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
	free(cpu_sizes);
}




void  GPUConvEngine::process(float* in) {

	//copy content and transfer
	cudaMemcpyToSymbol(INPUT, in, bs * sizeof(float));




	//launch the convolution Engine
	launchEngine();


	//perform overlap add
	for (int i = 0; i < bs; i++) {
		h_result_ptr[i] = (h_ConvolutionRes[i] + h_Overlap[i]) * 0.015;
		h_Overlap[i] = h_ConvolutionRes[i + bs - 1];

	}


}




void  GPUConvEngine::launchEngine() {
			shiftAndInsertKernel << <numBlocks, threadsPerBlock >> > (d_TimeDomain_padded);
	
			cudaError_t err = cudaGetLastError();
			// Launch the partitionedSubConvolution kernels in different streams
	 
			shared_partitioned_convolution1 << <dBlocks,dThreads , SHMEM,streams[0] >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
			shared_partitioned_convolution2 << <dBlocks, dThreads, SHMEM, streams[1] >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
			shared_partitioned_convolution3 << <dBlocks, dThreads, SHMEM, streams[2] >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
			shared_partitioned_convolution4 << <dBlocks, dThreads, SHMEM, streams[3] >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded);
			checkCudaError(cudaGetLastError(), "partitionedSubConvolution launch");
 
	 
		 
 
	// Copy the convolution results back to the host
	checkCudaError(cudaMemcpyAsync(h_ConvolutionRes, d_ConvolutionRes, h_convResSize * sizeof(float), cudaMemcpyDeviceToHost, streams[numOfSubPartitions - 1]), "d_ConvolutionRes copy to host");
	//set the result to 0

	cudaDeviceSynchronize();
	cudaMemset(d_ConvolutionRes, 0, h_convResSize * sizeof(float));
}
