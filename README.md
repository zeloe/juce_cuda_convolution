# juce_cuda_convolution
 Linear Convolution using CUDA 
 ```shell
  git clone https://github.com/zeloe/juce_cuda_convolution.git
  cd juce_cuda_convolution
  cd modules
  git clone https://github.com/juce-framework/JUCE.git
  cd ..
  cmake . -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -Bbuild
  cd build
  make -j20
  ./TestCUDA_artefacts/TestCUDA
```
# how it works
It performs a linear convolution on two different files. \
I provided a 128 channel impulse response and the resulting file is a 128 channel audio file. \
Enter file paths in terminal.




   \
  Note: \
  This only works on Ubuntu. \
  It still needs optimisation.
