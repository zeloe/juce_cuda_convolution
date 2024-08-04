
#include "GPUConvEngine.cuh"

__global__ void  partitionedSubConvolution(float* Result, const float* TimeDomainBuffer, const float* ImpulseResponse, const int bufferSize, const int impulseSize, const int begin) {
	int tidx = threadIdx.x + blockIdx.x * blockDim.x; // Global thread index
	int tidy = threadIdx.y + blockIdx.y * blockDim.y; // Global thread index
	 
	int convolutionResultSize = 2 * bufferSize - 1;
	//total partitions
	int totalPartitions = impulseSize / bufferSize;

	if (tidx < totalPartitions && tidy < convolutionResultSize) {
		//current partition
		int partitionIDX = tidx;
		//current position of slice
		int index_within_partition = tidy;
		//reset sum for each partitioned convolution
		float sum = 0.0f;

		// this is like a nested for loop to get correct convolution index for each partition
		// start index goes from 0 to buffersize
		//start index doesn't go below 0
		//start index slides from left to right
		int start_idx = max(0, index_within_partition - bufferSize + 1);
		//end index goes from buffersize to 0
		//defines the right bondary of the sliding window
		int end_idx = min(index_within_partition + 1, bufferSize);
		//for example 2 arrays of size 2 = (0,0) (0,1) (1,0) (1,1)
		//these index are constant for each convolution
		
		//this loop gets all the other index 
		for (int j = start_idx; j < end_idx; j++) {
			//flipping the filter
			int k = index_within_partition - j;
			//convolution 
			sum += (TimeDomainBuffer[j + partitionIDX * bufferSize + begin] * ImpulseResponse[k + partitionIDX * bufferSize + begin ]);
		}

		// Use atomicAdd to accumulate results correctly across threads
		atomicAdd(&Result[index_within_partition], sum);
	}
}

__global__ void  shiftAndInsertKernel(float* delayBuffer, const float* inputBuffer, const int blockSize, const int paddedSize) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Shift elements within the delay buffer
	if (tid < paddedSize - blockSize) {
		delayBuffer[tid + blockSize] = delayBuffer[tid];
	}

	// Insert new elements at the beginning of the delay buffer
	if (tid < blockSize) {
		delayBuffer[tid] = inputBuffer[tid];
	}
}

GPUConvEngine::GPUConvEngine() {

}



GPUConvEngine::~GPUConvEngine() {
	for(int i = 0; i < numOfSubPartitions; i++) {
	cudaStreamDestroy(streams[i]);
	}
	 
	cudaFree(d_IR_padded);
	 
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

	bs = maxBufferSize;
	h_result_ptr = (float*)calloc(bs, sizeof(float));
	h_ConvolutionRes = (float*)calloc(h_convResSize, sizeof(float));
	h_Overlap = (float*)calloc(bs, sizeof(float));
	 

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
	 
	 
	printf("Number of operations for each audioblock ~ %d\n", h_paddedSize * h_paddedSize);

	checkCudaError(cudaMalloc((void**)&d_IR_padded, h_paddedSize * sizeof(float)), "d_IR_padded malloc");
	checkCudaError(cudaMemset(d_IR_padded, 0, h_paddedSize * sizeof(float)), "d_IR_padded memset");
	checkCudaError(cudaMemcpy(d_IR_padded, impulseResponseBufferData, impulseResponseSize * sizeof(float), cudaMemcpyHostToDevice), "d_IR_padded memcpy");

	checkCudaError(cudaMalloc((void**)&d_TimeDomain_padded, h_paddedSize * sizeof(float)), "d_TimeDomain_padded malloc");
	checkCudaError(cudaMemset(d_TimeDomain_padded, 0, h_paddedSize * sizeof(float)), "d_TimeDomain_padded memset");

	for (int i = 0; i < numOfSubPartitions; i++) {
		checkCudaError(cudaStreamCreate(&streams[i]), "stream create");
	}
	dThreads.x = 32;
	dThreads.y = 32;
	dBlocks.x = (h_SizeOfSubPartitions / bs + dThreads.x - 1) / dThreads.x;
	dBlocks.y = (h_convResSize + dThreads.y - 1) / dThreads.y;
	threadsPerBlock.x = 1024;
	numBlocks.x = (h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x;
}




void  GPUConvEngine::process(float* in) {

	//copy content and transfer
	cudaMemcpy(d_Input, in, bs * sizeof(float), cudaMemcpyHostToDevice);

	 
	 

	//launch the convolution Engine
	launchEngine();

	
	//perform overlap add
	for (int i = 0; i < bs; i++) {
		h_result_ptr[i] = (h_ConvolutionRes[i] + h_Overlap[i]) * 0.015;
		h_Overlap[i] = h_ConvolutionRes[i + bs - 1];
		 
	}
	 
		 
}




void  GPUConvEngine::launchEngine(){
	shiftAndInsertKernel << <numBlocks, threadsPerBlock >> > (d_TimeDomain_padded, d_Input, bs, h_paddedSize);
	cudaError_t err = cudaGetLastError();
	// Launch the partitionedSubConvolution kernels in different streams
	for (int i = 0; i < numOfSubPartitions; i++) {
		partitionedSubConvolution << <dBlocks, dThreads, 0, streams[i] >> > (d_ConvolutionRes, d_TimeDomain_padded, d_IR_padded, bs, h_paddedSize, h_sizesOfSubPartitions[i]);
		checkCudaError(cudaGetLastError(), "partitionedSubConvolution launch");
	}
	// Synchronize all streams to ensure all work is completed
	for (int i = 0; i < numOfSubPartitions; i++) {
		checkCudaError(cudaStreamSynchronize(streams[i]), "cudaStreamSynchronize");
	}
	// Copy the convolution results back to the host
	checkCudaError(cudaMemcpyAsync(h_ConvolutionRes, d_ConvolutionRes, h_convResSize * sizeof(float), cudaMemcpyDeviceToHost, streams[numOfSubPartitions - 1]), "d_ConvolutionRes copy to host");
	//set the result to 0
	cudaMemset(d_ConvolutionRes, 0, h_convResSize * sizeof(float));
}
 