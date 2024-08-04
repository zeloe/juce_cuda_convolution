# juce_cuda_convolution
 Linear Convolution using CUDA 
 ```shell
  git clone https://github.com/zeloe/juce_cuda_convolution.git
  cmake -B build
  cd build
  cmake --build . --config Release -j24
```
# How it works
It performs time domain convolution on two different files on a realtime thread. 
## Index Calulation
Two arrays of size = 2 \
TimeDomainBuffer = (a0,a1) \
ImpulseResponse = (b0,b1)\
ConvolutionResult = (0,0,0) \
start_idx = max(index_within_partition, index_within_partition - size + 1)\
end_idx = min(index_within_partition + 1, size) \
flip index = index_within_partition - current_loop_index \
*index* = (index in time domain buffer, index in impulse response) 
### First Loop
index_within_partition  = 0 \
start_idx = max(0, 0 - 2 + 1) = 0 \
end_idx = min(0 + 1, 2) = 1 \
loop start_idx = 0 to end_idx = 1 \
flip index (1-1) 

*index* = (0,0) 

Res[0] += a0 * b0
### Second Loop
index_within_partition  = 1 \
start_idx = max(0, 1 - 2 + 1) = 0 \
end_idx = min(1 + 1, 2) = 2 \
loop start_idx = 0 to end_idx = 2\
flip index (1-0) \
*index* = (0,1) \
flip index (1-1) \
*index* = (1,0) 

Res[1] += a0 * b1 + a1 * b0

### Third Loop 
index_within_partition  = 2 \
start_idx = max(0, 2 - 2 + 1) = 2 \
end_idx = min(2 + 1, 2) = 2 \
loop start_idx = 2 to end_idx = 2\
flip index (2-1) \
*index* = (1,1) 

Res[2] += a1 * b1 

#### The two arrays need to have same size

## Time Domain Buffer
This holds all values for convolution in size of the padded impulseresponse. \
Padded impulse response is a multiple of buffersize and number of paralell convolutions. \ 
Insert and shift kernel copies new buffer at beggining. \
All other content gets shifted by buffersize. \
Content at end of Time Domain Buffer gets discarded. 

## Cuda
It uses four kernels in parallel using streams. \
Each of these kernels computes convolution results and sums them inside same buffer. 

## Hardware 
GeForce GTX 1660 Ti




# Note: 
It still needs optimisation. \
As a template repo i used this [template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template/blob/main/CMakeLists.txt).
\
Smaller Buffer Sizes = Lower Cpu usage \
Cuda == 12.3 \
VS = 2022 
