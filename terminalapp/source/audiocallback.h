#ifndef _AUDIOCALLBACK_H_

#define _AUDIOCALLBACK_H_



#include <JuceHeader.h>
#include "cuda/kernel.h"

    class MyAudioCallback : public juce::AudioIODeviceCallback
    {
    public:
        MyAudioCallback() {};
        ~MyAudioCallback() override
        {
            cudaFree(d_wetBuffer);
            cudaFree(d_dryBuffer);
            cudaFree(d_impBuffer);
        };
        void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
            int	numInputChannels,
            float* const* outputChannelData,
            int	numOutputChannels,
            int	numSamples,
            const AudioIODeviceCallbackContext& context) override
        {
            auto dry = tempDry.getWritePointer(0);
            auto imp = tempImp.getWritePointer(0);



            for (int i = 0; i < bs; i++) {
                dry[i] = dryPtr[i];
                imp[i] = impPtr[i];
               
                counter++;
                if (counter >= maxSize){
                    hasFinished = true;
                    break;
                }
                    
            }
            float* tempDryPtr = tempDry.getWritePointer(0);
            float* tempImpPtr = tempImp.getWritePointer(0);

            cudaMemcpy(d_dryBuffer, &tempDryPtr, sizeof(float) * bs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_impBuffer, &tempImpPtr, sizeof(float) * bs, cudaMemcpyHostToDevice);


            gpuRev(d_dryBuffer, d_impBuffer, bs, d_wetBuffer, d_size);
            float* tempWetPtr = tempWet.getWritePointer(0);
            cudaMemcpy(tempWetPtr, d_wetBuffer, sizeof(float) * bs, cudaMemcpyDeviceToHost);
            // Get the pointer to the output channel data
              float* outputDataL = outputChannelData[0];
              float* outputDataR = outputChannelData[1];
            for (int i = 0; i < bs; i++) {
                outputDataL[i] = tempWetPtr[i];
                outputDataR[i] = tempWetPtr[i];



            }
        
        }

        void prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize)
        {
            hasFinished = false;
            counter = 0;
            tempDry.setSize(1, bufferSize);
            tempDry.clear();
            tempWet.setSize(1, bufferSize);
            tempWet.clear();
            tempImp.setSize(1, bufferSize);
            tempImp.clear();
            dryPtr = dry.getWritePointer(0);
            impPtr = imp.getWritePointer(0);
            bs = bufferSize;
            maxSize = imp.getNumSamples();
           


             

            cudaMalloc((void**)&d_size, sizeof(int));

             

            float* d_scale;
            float h_scale = 0.15;
            cudaMalloc((void**)&d_scale, sizeof(float));
            cudaMemcpy(d_scale, &h_scale, sizeof(float), cudaMemcpyHostToDevice);

            // Copy to GPU
            cudaMemcpy(d_size, &bufferSize, sizeof(int), cudaMemcpyHostToDevice);
             


            cudaMalloc((void**)&d_wetBuffer, bufferSize * sizeof(float));
            cudaMalloc((void**)&d_dryBuffer, bufferSize * sizeof(float));
            cudaMalloc((void**)&d_impBuffer, bufferSize * sizeof(float));

            


        }
        bool hasFinished = false;


        virtual void 	audioDeviceAboutToStart(AudioIODevice* device) override {};
           
        virtual void 	audioDeviceStopped() override {};
            

        virtual void 	audioDeviceError(const String& errorMessage) override {};
            




    private:
        int bs = 0;
        int maxSize = 0;
        int counter = 0;
        float* dryPtr = nullptr;
        float* impPtr = nullptr;
        juce::AudioBuffer<float> tempDry;
        juce::AudioBuffer<float> tempImp;
        juce::AudioBuffer<float> tempWet;
        float* d_wetBuffer = nullptr;
        float* d_dryBuffer = nullptr;
        int* d_size = nullptr;
        float* d_impBuffer = nullptr;
    };

#endif