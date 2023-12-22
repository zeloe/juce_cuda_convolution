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

   \
  Note: \
  This only works on Ubuntu. \
  Edit paths to audiofiles in main.cpp
  
