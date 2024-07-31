
#include "GPUConvEngine.cuh"


GPUConvEngine::GPUConvEngine() {

}



GPUConvEngine::~GPUConvEngine() {
 
}
 

GPUConvEngine::GPUConvEngine(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize) {
	// get size for paddded host buffer
	
	twobs = maxBufferSize * 2 -1;

	//get the actual bufferSize
	bs = maxBufferSize;

	//prepare the overlapBuffer
	h_overLapBuffer.resize(bs);

	//fill it with 0
	thrust::fill(h_overLapBuffer.begin(), h_overLapBuffer.end(), 0);

	// get the number of partitions
	h_numPartitions = (int((impulseResponseSize) / bs) + 1);
	h_paddedSize = h_numPartitions * bs;
	trackSize = h_paddedSize;
	// create the h_impPaddedBuffer and a tempBuffer
	thrust::host_vector<float> tempBuffer(h_paddedSize);

	// Fill the vectors with zeros
	thrust::fill(tempBuffer.begin(), tempBuffer.end(), 0);

	//Fill the h_impPaddedBuffer with the impulseResponseBufferData
	// Copy the impulseResponseBufferData to tempBuffer
	thrust::copy(impulseResponseBufferData, impulseResponseBufferData + impulseResponseSize, tempBuffer.begin());


	// resize the device vector
	d_impPaddedBuffer.resize(h_paddedSize);

	//copy to gpu
	d_impPaddedBuffer = tempBuffer;

	//prepare the actual buffers for processing (host and device buffer)
	h_dryPaddedBuffer.resize(bs);
	thrust::fill(h_dryPaddedBuffer.begin(), h_dryPaddedBuffer.end(), 0);
	d_dryPaddedBuffer.resize(bs);
	thrust::fill(d_dryPaddedBuffer.begin(), d_dryPaddedBuffer.end(), 0);
	h_ressize = h_paddedSize;
	//prepare the accumBuffer
	d_accumBuffer.resize(h_ressize);

	//fill it with 0
	thrust::fill(d_accumBuffer.begin(), d_accumBuffer.end(), 0);

	//prepare sliceBuffers (host and device)
	d_sliceBuffer.resize(twobs);
	h_sliceBuffer.resize(twobs);

	//
	d_dryTrackBuffer.resize(trackSize);
	thrust::fill(d_dryTrackBuffer.begin(), d_dryTrackBuffer.end(), 0);

	//fill both with 0
	thrust::fill(d_sliceBuffer.begin(), d_sliceBuffer.end(), 0);
	thrust::fill(h_sliceBuffer.begin(), h_sliceBuffer.end(), 0);

	offset2 = bs;
	 
	offset = 0;
	h_result.resize(twobs);
	thrust::fill(h_result.begin(), h_result.end(), 0);
	h_result_ptr = thrust::raw_pointer_cast(h_result.data());
	d_delayBuffer.resize(h_paddedSize);
	 
	cudaStreamCreate(&stream);
	dim3 dThreads(16, 16);
	dim3 dBlocks((h_paddedSize / bs + dThreads.x - 1) / dThreads.x, (twobs + dThreads.y - 1) / dThreads.y);
	dim3 threadsPerBlock(256);
	dim3 numBlocks((h_paddedSize + threadsPerBlock.x - 1) / threadsPerBlock.x);
	convData = { d_delayBuffer.data().get() , d_impPaddedBuffer.data().get(),  d_accumBuffer.data().get(),bs,h_paddedSize,dThreads,dBlocks,d_sliceBuffer.data().get(),threadsPerBlock,numBlocks,stream,tempScale };
	writePointer = 0;
	d_tempBuffer.resize(h_paddedSize);
	readPointer = 0;
	tempScale = 1.f / h_paddedSize;
	std::cout << tempScale << std::endl;

}



void  GPUConvEngine::process(float* in) {

	//copy content
	thrust::copy(in, in + bs, h_dryPaddedBuffer.begin());

	//transfer to gpu
	d_dryPaddedBuffer = h_dryPaddedBuffer;
	 
	// Perform the delay line operation for the first buffer
	shiftAndInsert(d_delayBuffer, d_dryPaddedBuffer, bs, h_paddedSize, d_tempBuffer);

	//launch the convolution Engine
	this->launchEngine();




	// Copy to device host
	h_sliceBuffer = d_sliceBuffer;


	//add previous overlap to current overlap and write current content to output
	float* h_res = thrust::raw_pointer_cast(h_sliceBuffer.data());
	float* h_overlap = thrust::raw_pointer_cast(h_overLapBuffer.data());


	for (int i = 0; i < bs; i++) {
		h_result_ptr[i] = (h_res[i] +h_overlap[i]) * 0.015;
		h_overlap[i] = h_res[i + bs - 1];
		
	}
	thrust::fill(d_sliceBuffer.begin(), d_sliceBuffer.end(), 0);
		 
}




void  GPUConvEngine::launchEngine(){

	 


	gpuRev2(convData);
	
}

float* returnResult(float* result) {
	return result;
}