#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include "ConvKernel.cuh"



#ifndef _GPUConvEngine_H_

#define _GPUConvEngine_H_

template <typename T>
struct scale
{
	__host__ __device__
		T operator()(const T& x) const {
		return x * 0.5;
	}
}; 

class GPUConvEngine {
public:
	GPUConvEngine();
	~GPUConvEngine();

	GPUConvEngine(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize);
	
	void  process(float* in);
	 
	float* h_result_ptr = nullptr;
private:
	void   launchEngine();

	thrust::host_vector<float> h_overLapBuffer;
	thrust::host_vector<float> h_dryPaddedBuffer;
	thrust::host_vector<float> h_resultPaddedBuffer;
	thrust::host_vector<float> h_sliceBuffer;
	thrust::host_vector<float> h_result;
	thrust::device_vector<float> d_dryPaddedBuffer;
	thrust::device_vector<float> d_dryTrackBuffer;
	thrust::device_vector<float> d_impPaddedBuffer;
	thrust::device_vector<float> d_accumBuffer;
	thrust::device_vector<float> d_sliceBuffer;
	thrust::device_vector<float> d_delayBuffer;
	thrust::device_vector<float> d_tempBuffer;
	int twobs = 0;
	int bs = 0;
	int h_numPartitions = 0;
	int h_paddedSize = 0;
	int h_ressize = 0;
	int offset = 0;
	int offset2 = 0;
	int resultSize = 0;
	int trackSize = 0;
	// setup arguments
	scale<float> d_scale;
	ConvData convData;
	int* d_drySize = nullptr;
	int* d_wetSize = nullptr;
	cudaStream_t stream;
	int writePointer;
	int size;
	int delay;
	int readPointer; 
	float tempScale;


	// Function to shift elements in the delay line buffer
	void shiftAndInsert(thrust::device_vector<float>& delayBuffer, const thrust::device_vector<float>& inputBuffer, int blockSize, int paddedSize, thrust::device_vector<float>& tempBuffer) {

		// Copy elements from the delay buffer to the temporary buffer with the offset
		thrust::copy(delayBuffer.begin(), delayBuffer.end() - blockSize, tempBuffer.begin() + blockSize);
		thrust::copy(delayBuffer.end() - blockSize, delayBuffer.end(), tempBuffer.begin());

		// Insert the new elements at the beginning of the delay buffer
		thrust::copy(inputBuffer.begin(), inputBuffer.end(), delayBuffer.begin());

		// Copy back the shifted elements from the temporary buffer to the delay buffer
		thrust::copy(tempBuffer.begin() + blockSize, tempBuffer.end(), delayBuffer.begin() + blockSize);
	}


};



#endif