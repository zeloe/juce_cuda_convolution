# juce_cuda_convolution
Linear Convolution using CUDA
Clone the repository and build using CMake:
```shell
git clone https://github.com/zeloe/juce_cuda_convolution.git
cmake . -B build -G "Visual Studio 17 2022"
```
## How It Works
This project performs time-domain convolution on two different files in real-time.

Time Domain Buffer
This buffer holds all values for convolution, sized to the padded impulse response.
The padded impulse response is a multiple of the buffer size and the number of parallel convolutions.

Insert and shift the kernel copies into the new buffer at the beginning.

All other content gets shifted by the buffer size.

Content at the end of the Time Domain Buffer gets discarded.

## Hardware 
Tested on GeForce GTX 1660 Ti.




# Note: 
It still needs optimisation. \
Check out [here](https://github.com/zeloe/RTConvolver) VST3 Plugin to use in DAW. \
As a template repo i used this [template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template/blob/main/CMakeLists.txt).
\
Cuda == 12.3.52 \
MSVC == 19.36.32537.0
