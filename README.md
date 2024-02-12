# juce_cuda_convolution
 Linear Convolution using CUDA 
 ```shell
  git clone https://github.com/zeloe/juce_cuda_convolution.git
  cmake -B build
  cd build
  cmake --build . --config Release -j24
  ./CUDATemplate_artefacts/Release/CUDATemplate
```
# How it works
It performs a linear convolution on two different files. \
They are in project as binary data. \
I provided a 128 channel impulse response and the resulting file is a 128 channel audio file. 

# Note: 
This only works on Windows and Ubuntu. \
It still needs optimisation. \
There is still a bug when writing the file but it works :) \
As a template repo i used this [template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template/blob/main/CMakeLists.txt).
