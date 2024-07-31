
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
	 
	// Create the AudioDeviceManager instance
	juce::AudioDeviceManager audioDeviceManager;

	// Initialize the AudioDeviceManager with no input/output channels (default setup)
	audioDeviceManager.initialise(0, 2, nullptr, true);

	// Get the available device types
	auto& deviceTypes = audioDeviceManager.getAvailableDeviceTypes();
	
	// Iterate through the available device types
	for (auto& deviceType : deviceTypes) {
		std::cout << "Device Type: " << deviceType->getTypeName() << std::endl;
		if (deviceType->getTypeName() == "ASIO") {
			std::cout << "ASIO devices available." << std::endl;
		}
	}
	
	// Set the current audio device type to ASIO
	audioDeviceManager.setCurrentAudioDeviceType("DirectSound", true);
	// Get the current audio device
	auto* currentDevice = audioDeviceManager.getCurrentAudioDevice();
	std::cout << "Current Device ";
	std::cout << currentDevice->getDefaultBufferSize() << " = Current Buffer Size" << std::endl;
	std::cout << currentDevice->getTypeName() << std::endl;
	if (currentDevice == nullptr) {
		std::cerr << "No current audio device available!" << std::endl;
		return 1;
	}
	

	// Retrieve the current device setup
	juce::AudioDeviceManager::AudioDeviceSetup deviceSetup;
	audioDeviceManager.getAudioDeviceSetup(deviceSetup);

	// Set the desired buffer size (e.g., 128 samples)
	deviceSetup.bufferSize = 512;

	// Apply the updated setup
	juce::String error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);

	// Verify the buffer size has been set
	currentDevice = audioDeviceManager.getCurrentAudioDevice();

	if (currentDevice != nullptr) {
		std::cout << "New Buffer Size: " << currentDevice->getCurrentBufferSizeSamples() << " samples" << std::endl;
	}
	else {
		std::cerr << "No current audio device available after setting buffer size!" << std::endl;
		return 1;
	}
	
	std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>(bufferimp.getWritePointer(0), deviceSetup.bufferSize,bufferimp.getNumSamples(),bufferdry.getWritePointer(0),bufferdry.getNumSamples());
	juce::Thread::sleep(1000); // Sleep for 1 second

	audioDeviceManager.addAudioCallback(audiocallback.get());
	std::cout << "STARTING" << std::endl;
	while (true) {
	 
		// Print CPU usage
		//std::cout << "CPU Usage: " << aman.getCpuUsage() << "%" << std::endl;

		 
		 

		// Wait for a short duration before printing CPU usage again
		//std::this_thread::sleep_for(std::chrono::seconds(1)); // Adjust the duration as needed
	}

	 
	audioDeviceManager.removeAudioCallback(audiocallback.get());
	audioDeviceManager.closeAudioDevice();
 
	 
    return 0;
}

