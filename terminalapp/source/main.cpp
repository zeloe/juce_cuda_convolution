
#include "audiocallback.h"

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
	
 
	juce::ScopedJuceInitialiser_GUI juceInitialiser;
	 
	juce::AudioDeviceManager aman;
	auto& devices = aman.getAvailableDeviceTypes();
	 
	// Iterate through the available devices
	for (auto& device : devices) {
		std::cout << "Device Type: " << device->getTypeName() << std::endl;
		}
	
	juce::AudioDeviceManager::AudioDeviceSetup setup;
	setup.bufferSize = 512;
	setup.sampleRate = 44100;
	setup.inputChannels = 0;
	setup.outputChannels = 2;
	aman.setAudioDeviceSetup(setup, true);
	aman.setCurrentAudioDeviceType("DirectSound", true);

	auto device = aman.getCurrentAudioDevice();
	
	if (device == nullptr)
	{
		std::cerr << "No current audio device available!" << std::endl;
		return 1;
	}


	// Create an instance of MyAudioCallback
	std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>();

	
	audiocallback->prepare(bufferdry, bufferimp, device->getCurrentBufferSizeSamples());
	juce::Thread::sleep(1000); // Sleep for 1 second

	// Add audiocallback to the AudioDeviceManager
	aman.addAudioCallback(audiocallback.get());

	// Wait for user input to exit the application
	std::cout << "Press Enter to exit..." << std::endl;
	std::cin.get();

	 
	aman.removeAudioCallback(audiocallback.get());
	aman.closeAudioDevice();
 
	 
    return 0;
}

