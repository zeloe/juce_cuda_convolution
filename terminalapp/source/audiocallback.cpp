 

#include "audiocallback.h"

MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize)  {
    engine = std::make_unique<GPUConvEngine>(impulseResponseBufferData, maxBufferSize, impulseResponseSize);
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
    bs = maxBufferSize;
    this->dryPtr = dryPtr;
    this->drySize = drySize;
     

}
MyAudioCallback::~MyAudioCallback()  
    {

        
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
        engine->process(dry, outputChannelData);

        

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



     