 
#include "audiocallback.h"
#include <iostream>
#include <string>
#include <vector>

<<<<<<< Updated upstream
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
=======
// Convert juce::StringArray to std::vector<juce::String>
std::vector<juce::String> convertToVector(const juce::StringArray& stringArray) {
    return std::vector<juce::String>(stringArray.begin(), stringArray.end());
>>>>>>> Stashed changes
}

// Function to select device
int selectDevice(const std::vector<juce::String>& deviceNames, const std::string& prompt) {
    std::cout << prompt << std::endl;
    for (size_t i = 0; i < deviceNames.size(); ++i) {
        std::cout << i << ": " << deviceNames[i] << std::endl;
    }
    int selectedIndex;
    std::cin >> selectedIndex;
    return selectedIndex;
}

int main() {
    // Ask user to select a LiveInput
    std::cout << "With Live Input? Enter 1 for yes, 0 for no: ";
    bool liveInput;
    std::cin >> liveInput;

    const int bs = 256;
    const int channels = 1;

    juce::ScopedJuceInitialiser_GUI juceInitialiser;
    auto* IRStream = new juce::MemoryInputStream(BinaryData::imp_wav, BinaryData::imp_wavSize, false);
    juce::WavAudioFormat wavFormat;
    std::unique_ptr<juce::AudioFormatReader> impfile(wavFormat.createReaderFor(IRStream, false));
    int temp = impfile->lengthInSamples;
    std::cout << temp << " = Length in samples of impulse response" << std::endl;
    juce::AudioBuffer<float> bufferimp;
    bufferimp.setSize(channels, temp);
    bufferimp.clear();
    impfile->read(&bufferimp, 0, temp, 0, true, true);

    // Create the AudioDeviceManager instance
    juce::AudioDeviceManager audioDeviceManager;

    if (liveInput) {
        // Initialize the AudioDeviceManager with input and output channels
        juce::String error = audioDeviceManager.initialise(1, 2, nullptr, true);
        if (error.isNotEmpty()) {
            std::cerr << "Error initializing AudioDeviceManager: " << error << std::endl;
            return 1;
        }

        // Get the available device types
        auto& deviceTypes = audioDeviceManager.getAvailableDeviceTypes();

        // Iterate through the available device types and print them
        std::cout << "Available Device Types:" << std::endl;
        for (size_t i = 0; i < deviceTypes.size(); ++i) {
            deviceTypes[i]->scanForDevices();
            std::cout << i << ": " << deviceTypes[i]->getTypeName() << std::endl;

            auto deviceNames = deviceTypes[i]->getDeviceNames();
            for (size_t j = 0; j < deviceNames.size(); ++j) {
                std::cout << "  " << j << ": " << deviceNames[j] << std::endl;
            }
        }

        // Convert juce::StringArray to std::vector<juce::String>
        auto inputDeviceNames = convertToVector(deviceTypes[selectedInputTypeIndex]->getDeviceNames());
        auto outputDeviceNames = convertToVector(deviceTypes[selectedOutputTypeIndex]->getDeviceNames());

        // Ask user to select an input device
        int selectedInputDeviceIndex = selectDevice(inputDeviceNames, "Select an input device index: ");
        if (selectedInputDeviceIndex < 0 || selectedInputDeviceIndex >= inputDeviceNames.size()) {
            std::cerr << "Invalid device index!" << std::endl;
            return 1;
        }

        // Ask user to select an output device
        int selectedOutputDeviceIndex = selectDevice(outputDeviceNames, "Select an output device index: ");
        if (selectedOutputDeviceIndex < 0 || selectedOutputDeviceIndex >= outputDeviceNames.size()) {
            std::cerr << "Invalid device index!" << std::endl;
            return 1;
        }

        juce::AudioDeviceManager::AudioDeviceSetup deviceSetup;
        audioDeviceManager.getAudioDeviceSetup(deviceSetup);

        deviceSetup.inputDeviceName = inputDeviceNames[selectedInputDeviceIndex];
        deviceSetup.outputDeviceName = outputDeviceNames[selectedOutputDeviceIndex];
        deviceSetup.bufferSize = bs;

        error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);
        if (error.isNotEmpty()) {
            std::cerr << "Error setting up audio device: " << error << std::endl;
            return 1;
        }

        auto* currentDevice = audioDeviceManager.getCurrentAudioDevice();
        if (currentDevice == nullptr) {
            std::cerr << "No current audio device available!" << std::endl;
            return 1;
        }

        std::cout << "Current Input Device: " << deviceSetup.inputDeviceName << std::endl;
        std::cout << "Current Output Device: " << deviceSetup.outputDeviceName << std::endl;
        std::cout << bs << " = Current Buffer Size" << std::endl;

        std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>(bufferimp.getWritePointer(0), temp, deviceSetup.bufferSize);
        juce::Thread::sleep(1000); // Sleep for 1 second

        audioDeviceManager.addAudioCallback(audiocallback.get());
        std::cout << "STARTING" << std::endl;
        while (true) {
            // Print CPU usage
            std::cout << "CPU Usage: " << audioDeviceManager.getCpuUsage() << "%" << std::endl;

            // Wait for a short duration before printing CPU usage again
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Adjust the duration as needed
        }

        audioDeviceManager.removeAudioCallback(audiocallback.get());
        audioDeviceManager.closeAudioDevice();
    }
    else {
        auto* DRYStream = new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
        const juce::int64 channels = impfile->numChannels;
        juce::WavAudioFormat wavFormat2;
        std::unique_ptr<juce::AudioFormatReader> dry(wavFormat2.createReaderFor(DRYStream, false));

        int temp2 = dry->lengthInSamples;

        juce::AudioBuffer<float> bufferdry;
        bufferdry.setSize(channels, temp2);
        bufferdry.clear();

        dry->read(&bufferdry, 0, temp2, 0, true, true);

        // Initialize the AudioDeviceManager with input and output channels
        juce::String error = audioDeviceManager.initialise(0, 2, nullptr, true);

        if (error.isNotEmpty()) {
            std::cerr << "Error initializing AudioDeviceManager: " << error << std::endl;
            return 1;
        }

        // Get the available device types
        auto& deviceTypes = audioDeviceManager.getAvailableDeviceTypes();

        // Iterate through the available device types and print them
        std::cout << "Available Device Types:" << std::endl;
        for (size_t i = 0; i < deviceTypes.size(); ++i) {
            deviceTypes[i]->scanForDevices();
            std::cout << i << ": " << deviceTypes[i]->getTypeName() << std::endl;

            auto deviceNames = deviceTypes[i]->getDeviceNames();
            for (size_t j = 0; j < deviceNames.size(); ++j) {
                std::cout << "  " << j << ": " << deviceNames[j] << std::endl;
            }
        }


        auto outputDeviceNames = convertToVector(deviceTypes[selectedOutputTypeIndex]->getDeviceNames());



        // Ask user to select an output device
        int selectedOutputDeviceIndex = selectDevice(outputDeviceNames, "Select an output device index: ");
        if (selectedOutputDeviceIndex < 0 || selectedOutputDeviceIndex >= outputDeviceNames.size() ){
            std::cerr << "Invalid device index!" << std::endl;
            return 1;
        }



        juce::AudioDeviceManager::AudioDeviceSetup deviceSetup;
            audioDeviceManager.getAudioDeviceSetup(deviceSetup);



            deviceSetup.outputDeviceName = outputDeviceNames[selectedOutputDeviceIndex];
        deviceSetup.bufferSize = bs;

        error = audioDeviceManager.setAudioDeviceSetup(deviceSetup, true);
        if (error.isNotEmpty()) {
            std::cerr << "Error setting up audio device: " << error << std::endl;
            return 1;
        }

        auto* currentDevice = audioDeviceManager.getCurrentAudioDevice();
        if (currentDevice == nullptr) {
            std::cerr << "No current audio device available!" << std::endl;
            return 1;
        }


        std::cout << "Current Output Device: " << deviceSetup.outputDeviceName << std::endl;
        std::cout << bs << " = Current Buffer Size" << std::endl;
        std::unique_ptr<MyAudioCallback> audiocallback = std::make_unique<MyAudioCallback>(bufferimp.getWritePointer(0), temp, deviceSetup.bufferSize, bufferdry.getWritePointer(0), temp2, liveInput);
        audioDeviceManager.addAudioCallback(audiocallback.get());
        std::cout << "STARTING" << std::endl;
        while (true)
        {
            // Print CPU usage
            std::cout << "CPU Usage: " << audioDeviceManager.getCpuUsage() << "%" << std::endl;
            // Wait for a short duration before printing CPU usage again
            std::this_thread::sleep_for(std::chrono::seconds(1)); // Adjust the duration as needed
        }
        audioDeviceManager.removeAudioCallback(audiocallback.get());
        audioDeviceManager.closeAudioDevice();
    }


   

    return 0;




}
