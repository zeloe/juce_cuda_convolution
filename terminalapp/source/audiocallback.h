#ifndef _AUDIOCALLBACK_H_

#define _AUDIOCALLBACK_H_



#include <JuceHeader.h>
#include "cuda/kernel.h"

    class MyAudioCallback : public juce::AudioIODeviceCallback, public juce::Thread
    {
    public:
        MyAudioCallback(): juce::Thread("FilterPartionThread") {};
        ~MyAudioCallback() override
        {

            cudaFree(d_wetBuffer);
            cudaFree(d_dryBuffer);
             
            stopThread(2000);
        };
        void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
            int	numInputChannels,
            float* const* outputChannelData,
            int	numOutputChannels,
            int	numSamples,
            const AudioIODeviceCallbackContext& context) override
        {
            //copy Slice
            auto dry = tempDry.getWritePointer(0);
             
            

            for (int i = 0; i < bs; i++) {
               
                
                    dry[i] = (dryPtr[counter]);
                    counter++;
                if (counter >= drySize) {
                    counter = 0;
                }
            }
            float* tempDryPtr = tempDry.getWritePointer(0);
         

            cudaMemcpy(d_dryBuffer, tempDryPtr, sizeof(float) * twobs, cudaMemcpyHostToDevice);
            

            
            gpuRev(d_dryBuffer, d_fistImpPart, twobs, d_wetBuffer, d_size);
            startThread(Priority::high);
            float* tempWetPtr = tempWet.getWritePointer(0);
            cudaMemcpy(tempWetPtr, d_wetBuffer, sizeof(float) * twobs, cudaMemcpyDeviceToHost);
            // Get the pointer to the output channel data
              float* outputDataL = outputChannelData[0];
              float* outputDataR = outputChannelData[1];
              float* accums = accumBuffer.getWritePointer(0);
              float* overLap = overLapBuffer.getWritePointer(0);
            if(first == true){
                for (int i = 0; i < bs; i++) {
                    outputDataL[i] = (tempWetPtr[i]  + overLap[i]);
                    outputDataR[i] = (tempWetPtr[i]  + overLap[i]);
                    overLap[i] = tempWetPtr[i + bs];
                     
                }
                first = false;
            }
            else {
                for (int i = 0; i < bs; i++) {
                    outputDataL[i] = (accums[counter2] + overLap[i]);
                    outputDataR[i] = (accums[counter2] + overLap[i]);
                    overLap[i] = (accums[counter2 + bs]);
                    counter2++;
                    if (counter2 > accumBuffer.getNumSamples() - bs)
                        break;
                }
            }
        
        }
        void run() override {

            if (!threadShouldExit()) {
                 

                partgpuRev(d_dryBuffer, d_otherParts, twobs, d_accumBuffer, d_size, d_numPartitions, h_numPartitions);
                auto accums = accumBuffer.getWritePointer(0);
                cudaMemcpy(accums, d_accumBuffer, sizeof(float) * paddedSize, cudaMemcpyDeviceToHost);




            }
        }
        void prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize)
        {
             
            counter = 0;
        
            //the overlapbuffer
            overLapBuffer.setSize(1, bufferSize);
            overLapBuffer.clear();
            
            twobs = bufferSize * 2;
            //set the size for input audio
            tempDry.setSize(1, twobs);
            tempDry.clear();
            tempWet.setSize(1, twobs);
            tempWet.clear();
            
            dryPtr = dry.getWritePointer(0);
            impPtr = imp.getWritePointer(0);
            bs = bufferSize;
            drySize = dry.getNumSamples();
            maxSize = imp.getNumSamples() + dry.getNumSamples();
            accumBuffer.setSize(1, maxSize);
            accumBuffer.clear();
            auto accumBufPtr = accumBuffer.getWritePointer(0);
            cudaMalloc((void**)&d_accumBuffer, maxSize * sizeof(float));
            cudaMemcpy(d_accumBuffer, accumBufPtr, maxSize * sizeof(float), cudaMemcpyHostToDevice);
            impFull.setSize(1, maxSize);
            //AllocateMemory
            cudaMalloc((void**)&d_twosize, sizeof(int));
            cudaMalloc((void**)&d_size, sizeof(int));
            cudaMalloc((void**)&slicePtrIn, twobs * sizeof(float));
            cudaMalloc((void**)&d_wetBuffer, twobs * sizeof(float));
            cudaMalloc((void**)&d_dryBuffer, twobs * sizeof(float));
            cudaMalloc((void**)&slicePtrOut, twobs * sizeof(float));
            cudaMalloc((void**)&d_impBuffer, twobs * sizeof(float));

            //copy first Part for filterPartitioning
            cudaMalloc((void**)&d_fistImpPart, twobs * sizeof(float));
            auto firstSlice = imp.getWritePointer(0);
            auto tempWetPtr = tempWet.getWritePointer(0);
            for(int i = 0; i < bs; i++){
                tempWetPtr[i] = firstSlice[i];
            }

            cudaMemcpy(d_fistImpPart, tempWetPtr, sizeof(float) * twobs, cudaMemcpyHostToDevice);
            tempWet.clear();
            
            //
            //copy other parts for filterPartitioning
            //resize the buffer to be in equal parts of the blocksize
            h_numPartitions = (int((imp.getNumSamples()) / twobs) + 1);
            paddedSize = h_numPartitions * twobs;
            cudaMalloc((void**)&d_numPartitions, sizeof(int));
            cudaMemcpy(d_numPartitions, &h_numPartitions, sizeof(int), cudaMemcpyHostToDevice);
            cudaMalloc((void**)&d_otherParts, (paddedSize) * sizeof(float));
            juce::AudioBuffer<float> otherPartsBuffer;
            otherPartsBuffer.setSize(1,paddedSize);
            otherPartsBuffer.clear();

            auto fParts = otherPartsBuffer.getWritePointer(0);
            auto fullImp = imp.getWritePointer(0);
            //allocateMemory for device AccumBuffer
            cudaMalloc((void**)&d_accumBuffer, (paddedSize) * sizeof(float));
            //fill it with 0 since the other buffer is cleared
            cudaMemcpy(d_accumBuffer, fParts, (paddedSize) * sizeof(float), cudaMemcpyHostToDevice);

            //copy remaning parts to the other buffer
            int track = paddedSize - twobs;
            for (int i = 0; i < twobs; i++)
            {
                fParts[i] = 0;
                //zero pad the buffer
                fParts[track+ i] = 0;
            }
            //pad the buffer
            //copy content
            for (int i = twobs; i < imp.getNumSamples(); i++) {
                fParts[i] = fullImp[i];
                
            }
            //allocate on gpu
            cudaMemcpy(d_otherParts, fParts, sizeof(float) * paddedSize, cudaMemcpyHostToDevice);
            //get a buffer filled of 0
            impFull.clear();
            auto fullConvImp = impFull.getWritePointer(0);
            cudaMemcpy(d_wetBuffer, fullConvImp, sizeof(float) * twobs, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dryBuffer, fullConvImp, sizeof(float) * twobs, cudaMemcpyHostToDevice);


          
            
            // Copy to GPU
            cudaMemcpy(d_size, &twobs, sizeof(int), cudaMemcpyHostToDevice);
             
           

            
            

             
            


        }
        bool hasFinished = false;


        virtual void 	audioDeviceAboutToStart(AudioIODevice* device) override {};
           
        virtual void 	audioDeviceStopped() override {};
            

        virtual void 	audioDeviceError(const String& errorMessage) override 
        {
            std::cout << errorMessage << std::endl;
        };
            

       


    private:
        int bs = 0;
        int twobs = 0;
        int maxSize = 0;
        int counter = 0;
        int counter2 = 0;
        float* dryPtr = nullptr;
        float* impPtr = nullptr;
        float* slicePtrOut = nullptr;
        float* slicePtrIn = nullptr;
        float* d_fistImpPart = nullptr;
        int* d_twosize = nullptr;
        float* d_otherParts = nullptr;
        juce::AudioBuffer<float> tempDry;
        juce::AudioBuffer<float> impFull;
        juce::AudioBuffer<float> tempWet;
        juce::AudioBuffer<float> accumBuffer;
        juce::AudioBuffer<float> overLapBuffer;
        float* d_accumBuffer = nullptr;
        float* d_wetBuffer = nullptr;
        float* d_dryBuffer = nullptr;
        int* d_size = nullptr;
        float* d_impBuffer = nullptr;
        int h_numPartitions = 0;
        int* d_numPartitions = nullptr;
        int paddedSize = 0;
        bool first = true;
        int drySize = 0;
    };

#endif