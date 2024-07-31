 

#include "audiocallback.h"

MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize) : juce::Thread("FilterPartionThread") {

    engine = std::make_unique<GPUConvEngine>(impulseResponseBufferData, maxBufferSize, impulseResponseSize);
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
    bs = maxBufferSize;
    this->dryPtr = dryPtr;
    this->drySize = drySize;
     

}
MyAudioCallback::~MyAudioCallback()  
    {

       

        stopThread(2000);
    };
 
void MyAudioCallback::audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
        int	numInputChannels,
        float* const* outputChannelData,
        int	numOutputChannels,
        int	numSamples,
        const AudioIODeviceCallbackContext& context)  
    {
        //copy Slice
        
        auto dry = tempDry.getWritePointer(0);



        for (int i = 0; i < numSamples; i++) {


            dry[i] = (dryPtr[counter]);
            counter++;
            if (counter >= drySize) {
                counter = 0;
            }
        }
        //startThread(Priority::normal);
        engine->process(dry);
        float* ptr_L = outputChannelData[0];
        float* ptr_R = outputChannelData[1];
        for (int i = 0; i < numSamples; i++) {
            ptr_L[i] = engine->h_result_ptr[i];
            ptr_R[i] = engine->h_result_ptr[i];

        }
        

    }
 void MyAudioCallback::run()   {
    float* dry = tempDry.getWritePointer(0);
    engine->process(dry);
  
    }
void MyAudioCallback::prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize)
    {

       









    }
   


      void 	MyAudioCallback::audioDeviceAboutToStart(AudioIODevice* device)   {};

      void 	MyAudioCallback::audioDeviceStopped()   {};


      void 	MyAudioCallback::audioDeviceError(const String& errorMessage)  
    {
        std::cout << errorMessage << std::endl;
    };



     