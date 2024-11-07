# juce_cuda_convolution
 Linear Convolution using CUDA 
 ```shell
  git clone https://github.com/zeloe/juce_cuda_convolution.git
  cmake . -B build -G "Visual Studio 17 2022"
```
# How it works
It performs time domain convolution on two different files on a realtime thread.  

## Time Domain Buffer
This holds all values for convolution in size of the padded impulseresponse. \
Padded impulse response is a multiple of buffersize and number of paralell convolutions. 
Insert and shift kernel copies new buffer at beggining. \
All other content gets shifted by buffersize. \
Content at end of Time Domain Buffer gets discarded. 

## Hardware 
GeForce GTX 1660 Ti




# Note: 
It still needs optimisation. \
Check out [here](https://github.com/zeloe/RTConvolver) VST3 Plugin to use in DAW. \
As a template repo i used this [template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template/blob/main/CMakeLists.txt).
\
Cuda == 12.3.52 \
MSVC == 19.36.32537.0
