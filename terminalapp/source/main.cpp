
#include "audiocallback.h"
#include <windows.h> // Include Windows headers for CoInitializeEx
#include <JuceHeader.h>
#include <iostream>
#include <string>
 

int main()
{
 
	auto* IRStream = new juce::MemoryInputStream(BinaryData::imp_wav, BinaryData::imp_wavSize,false);
	auto* DRYStream= new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
	
	WavAudioFormat wavFormat;
	std::unique_ptr<AudioFormatReader> impfile (wavFormat.createReaderFor (IRStream, false));
 
	
	const juce::int64 channels = impfile->numChannels;
	WavAudioFormat wavFormat2;
	std::unique_ptr<AudioFormatReader> dry (wavFormat2.createReaderFor (DRYStream, false));
 
	
	
	
	int temp = impfile->lengthInSamples;
	std::cout<<temp<< " Length in samples of impulse response" << std::endl;
	std::cout<<channels << " Total number of channels of impulse Response" << std::endl; 
	
	int temp2 = dry->lengthInSamples;
	
	std::cout<<temp2<< " Length in samples of dry audio" << std::endl; 
	
	temp = temp2 + temp;

	
	
	
	juce::AudioBuffer<float> bufferimp;
	juce::AudioBuffer<float> bufferdry;
	bufferdry.setSize(channels, temp);
	bufferdry.clear();
	bufferimp.setSize(channels, temp);
	bufferimp.clear();
	juce::AudioBuffer<float> bufferout;
	bufferout.setSize(channels,temp);
	bufferout.clear();

	impfile->read(&bufferimp, 0, temp, 0, true, true);
	 
	dry->read(&bufferdry, 0, temp, 0, true, true);
	
	HRESULT hr;
	hr = CoInitializeEx(0, COINIT_MULTITHREADED);
	juce::AudioDeviceManager aman;
	// Initialize audiocallback
	aman.initialiseWithDefaultDevices(0, 2);
	AudioIODevice* device = aman.getCurrentAudioDevice();
	

	// Create an instance of MyAudioCallback
	std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>();

	
	audiocallback->prepare(bufferdry, bufferimp, device->getCurrentBufferSizeSamples());

	// Add audiocallback to the AudioDeviceManager
	aman.addAudioCallback(audiocallback.get());

	while (!audiocallback->hasFinished)
	{
		juce::Thread::sleep(10);
	}
	 
	aman.removeAudioCallback(audiocallback.get());
	aman.closeAudioDevice();
	// When your program is about to exit, call CoUninitialize
	CoUninitialize();
    return 0;
}

