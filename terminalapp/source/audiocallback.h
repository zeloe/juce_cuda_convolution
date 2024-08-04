#ifndef _AUDIOCALLBACK_H_

#define _AUDIOCALLBACK_H_


#include "JuceHeader.h"
#include "cuda/GPUConvEngine.cuh"

    class MyAudioCallback : public juce::AudioIODeviceCallback 
    {
    public:
        MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize);
        ~MyAudioCallback() override;
        void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
            int	numInputChannels,
            float* const* outputChannelData,
            int	numOutputChannels,
            int	numSamples,
            const AudioIODeviceCallbackContext& context) override;
         
        void prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize);
        bool hasFinished = false;


        virtual void 	audioDeviceAboutToStart(AudioIODevice* device) override;
           
        virtual void 	audioDeviceStopped() override;
            

        virtual void 	audioDeviceError(const String& errorMessage) override;
            

       


    private:
        std::unique_ptr<GPUConvEngine> engine;
        int bs = -1;
        int drySize = -1;
        int counter = 0;
        juce::AudioBuffer<float>tempDry;
        float* dryPtr = nullptr;
        float* const* out = nullptr;
        float* in= nullptr;
        bool isThreadRunning = false;
        CriticalSection bufferMutex;
    };

#endif