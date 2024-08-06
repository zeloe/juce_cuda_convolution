#ifndef _AUDIOCALLBACK_H_
#define _AUDIOCALLBACK_H_

#include <JuceHeader.h>
#include "cuda/GPUConvEngine.cuh"

<<<<<<< Updated upstream
    class MyAudioCallback : public juce::AudioIODeviceCallback, public juce::Thread
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
        void run() override;
        void prepare(juce::AudioBuffer<float>& dry, juce::AudioBuffer<float>& imp, int bufferSize);
        bool hasFinished = false;
=======
class MyAudioCallback : public juce::AudioIODeviceCallback
{
public:
    MyAudioCallback(float* impulseResponseBufferData, int impulseResponseSize, int maxBufferSize, float* dryPtr, int drySize, bool liveInput);
    MyAudioCallback(float* impulseResponseBufferData, int impulseResponseSize, int maxBufferSize, bool liveInput);
    ~MyAudioCallback() override;
>>>>>>> Stashed changes

    void audioDeviceIOCallbackWithContext(const float* const* inputChannelData,
        int numInputChannels,
        float* const* outputChannelData,
        int numOutputChannels,
        int numSamples,
        const juce::AudioIODeviceCallbackContext& context) override;

    void audioDeviceAboutToStart(juce::AudioIODevice* device) override;
    void audioDeviceStopped() override;
    void audioDeviceError(const juce::String& errorMessage) override;

private:
    std::unique_ptr<GPUConvEngine> engine;
    int bs = -1;
    int drySize = -1;
    int counter = 0;
    juce::AudioBuffer<float> tempDry;
    float* dryPtr = nullptr;
    float* const* out = nullptr;
    float* in = nullptr;
    bool isThreadRunning = false;
    bool input;
};

#endif // _AUDIOCALLBACK_H_
