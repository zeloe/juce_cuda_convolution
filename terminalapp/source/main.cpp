
#include "audiocallback.h"

#include <JuceHeader.h>
#include <iostream>
#include <string>
 

int main()
{
	const int bs = 256;
	auto* IRStream = new juce::MemoryInputStream(BinaryData::imp_wav, BinaryData::imp_wavSize,false);
	auto* DRYStream= new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
	
	WavAudioFormat wavFormat;
	std::unique_ptr<AudioFormatReader> impfile (wavFormat.createReaderFor (IRStream, false));
 
	
	const juce::int64 channels = impfile->numChannels;
	WavAudioFormat wavFormat2;
	std::unique_ptr<AudioFormatReader> dry (wavFormat2.createReaderFor (DRYStream, false));
 
	
	
	
	int temp = impfile->lengthInSamples;
	std::cout<<temp<< " =  Length in samples of impulse response" << std::endl;
	 
	
	int temp2 = dry->lengthInSamples;
	 
	 
	
	
	juce::AudioBuffer<float> bufferimp;
	juce::AudioBuffer<float> bufferdry;
	bufferdry.setSize(channels, temp2);
	bufferdry.clear();
	bufferimp.setSize(channels, temp);
	bufferimp.clear();
	 

	impfile->read(&bufferimp, 0, temp, 0, true, true);
	 
	dry->read(&bufferdry, 0, temp2, 0, true, true);
	
 
	juce::ScopedJuceInitialiser_GUI juceInitialiser;
	 
	// Create the AudioDeviceManager instance
	juce::AudioDeviceManager audioDeviceManager;

	// Initialize the AudioDeviceManager with no input/output channels (default setup)
	audioDeviceManager.initialise(0, 2, nullptr, true);

	// Get the available device types
	auto& deviceTypes = audioDeviceManager.getAvailableDeviceTypes();
	
	// Iterate through the available device types
	for (auto& deviceType : deviceTypes) {
		std::cout << "Device Type: " << deviceType->getTypeName() << std::endl;
	}
	
	
	// Get the current audio device
	auto* currentDevice = audioDeviceManager.getCurrentAudioDevice();
	std::cout << "Current Device ";
	std::cout << currentDevice->getTypeName() << std::endl;
	std::cout << bs << " = Current Buffer Size" << std::endl;

	if (currentDevice == nullptr) {
		std::cerr << "No current audio device available!" << std::endl;
		return 1;
	}
	

	// Retrieve the current device setup
	juce::AudioDeviceManager::AudioDeviceSetup deviceSetup;
	audioDeviceManager.getAudioDeviceSetup(deviceSetup);

	// Set the desired buffer size (e.g., 128 samples)
	deviceSetup.bufferSize = bs;

	// Apply the updated setup
	juce::String error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);

	// Verify the buffer size has been set
	currentDevice = audioDeviceManager.getCurrentAudioDevice();

	
	std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>(bufferimp.getWritePointer(0), deviceSetup.bufferSize,bufferimp.getNumSamples(),bufferdry.getWritePointer(0),bufferdry.getNumSamples());
	juce::Thread::sleep(1000); // Sleep for 1 second

	audioDeviceManager.addAudioCallback(audiocallback.get());
	std::cout << "STARTING" << std::endl;
	while (true) {
	 
		// Print CPU usage
		std::cout << "CPU Usage: " << audioDeviceManager.getCpuUsage() * 100 << " %" << std::endl;

		 
		 

		//Wait for a short duration before printing CPU usage again
		std::this_thread::sleep_for(std::chrono::seconds(1)); // Adjust the duration as needed
	}

	 
	audioDeviceManager.removeAudioCallback(audiocallback.get());
	audioDeviceManager.closeAudioDevice();
 
	 
    return 0;
}

