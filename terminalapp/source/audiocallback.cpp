#include "audiocallback.h"

<<<<<<< Updated upstream
MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize) : juce::Thread("FilterPartionThread") {

    engine = std::make_unique<GPUConvEngine>(impulseResponseBufferData, maxBufferSize, impulseResponseSize);
=======
MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int maxBufferSize, int impulseResponseSize, float* dryPtr, int drySize, bool liveInput)
    : engine(std::make_unique<GPUConvEngine>(impulseResponseBufferData, maxBufferSize, impulseResponseSize)),
    bs(maxBufferSize),
    dryPtr(dryPtr),
    drySize(drySize),
    input(liveInput)
{
>>>>>>> Stashed changes
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
}

<<<<<<< Updated upstream
       

        stopThread(2000);
    };
 
=======
MyAudioCallback::MyAudioCallback(float* impulseResponseBufferData, int impulseResponseSize, int maxBufferSize, bool liveInput)
    : engine(std::make_unique<GPUConvEngine>(impulseResponseBufferData, maxBufferSize, impulseResponseSize)),
    bs(maxBufferSize),
    input(liveInput)
{
    tempDry.setSize(1, maxBufferSize);
    tempDry.clear();
}

MyAudioCallback::~MyAudioCallback()
{
    // Cleanup resources if needed
}

MyAudioCallback::MyAudioCallback()
    : bs(-1), drySize(-1), counter(0), isThreadRunning(false), input(false)
{
    // Default constructor implementation
}

>>>>>>> Stashed changes
void MyAudioCallback::audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
    int numInputChannels,
    float* const* outputChannelData,
    int numOutputChannels,
    int numSamples,
    const juce::AudioIODeviceCallbackContext& context)
{
    if (inputChannelData == nullptr || outputChannelData == nullptr)
        return;

    auto dry = tempDry.getWritePointer(0);

    if (input)
    {
        for (int i = 0; i < numSamples; i++)
        {
            dry[i] = (inputChannelData[0][i]);
        }
    }
    else
    {
        for (int i = 0; i < numSamples; i++)
        {
            dry[i] = dryPtr[counter];
            counter++;
            if (counter >= drySize)
            {
                counter = 0;
            }
        }
<<<<<<< Updated upstream
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

       









=======
>>>>>>> Stashed changes
    }

    // Process the dry buffer
    engine->process(dry);

    // Get the output pointers
    float* ptr_L = outputChannelData[0];
    float* ptr_R = outputChannelData[1];

    for (int i = 0; i < numSamples; i++)
    {
        ptr_L[i] = engine->h_result_ptr[i];
        ptr_R[i] = engine->h_result_ptr[i];
    }
}

void MyAudioCallback::audioDeviceAboutToStart(juce::AudioIODevice* device)
{
    // Implementation when audio device is about to start
}

void MyAudioCallback::audioDeviceStopped()
{
    // Implementation when audio device stops
}

void MyAudioCallback::audioDeviceError(const juce::String& errorMessage)
{
    std::cout << errorMessage << std::endl;
}
