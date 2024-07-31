# juce_cuda_convolution
 Linear Convolution using CUDA 
 ```shell
  git clone https://github.com/zeloe/juce_cuda_convolution.git
  cmake -B build
  cd build
  cmake --build . --config Release -j24
```
# How it works
It performs time domain convolution on two different files on a realtime thread. \
They are in project as binary data. \
I used uniformly filter partition using overlap add. 
# Note: 
This only works on Windows and Ubuntu. \
It still needs optimisation. \
As a template repo i used this [template](https://github.com/anthonyalfimov/JUCE-CMake-Plugin-Template/blob/main/CMakeLists.txt).
\
will add more detailed description soon 
