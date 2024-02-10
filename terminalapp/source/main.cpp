#include "cuda/kernel.h"
#include <JuceHeader.h>
#include <iostream>
#include <string>


int main()
{
 
	std::string drypath;
	std::string irpath;
	std::string output;
	std::cout << "Path to Dry File= " << std::endl;
	std::cin >> drypath; 
	std::cout << "Path to Impulse Response File= " << std::endl;
	std::cin >> irpath;
	std::cout << "Output filepath " << std::endl;
	std::cin >> output;

	juce::AudioFormatManager formatManager;
	output = "/home/zelo/Desktop/";
	formatManager.registerBasicFormats();
	juce::File file = juce::File(irpath);
	auto* impfile = formatManager.createReaderFor(file);
	const unsigned int channels = impfile->numChannels;
	
	
	juce::AudioFormatManager formatManager2;
	formatManager2.registerBasicFormats();
	juce::File file2 = juce::File(drypath);
	auto* dry = formatManager2.createReaderFor(file2);
	
	
	
	int temp = impfile->lengthInSamples * 2;
	std::cout<<temp / 2 << " Length in samples of impulse response" << std::endl;
	std::cout<<channels << " Total number of channels of impulse Response" << std::endl; 
	
	
	
	
	int temp2 = dry->lengthInSamples * 2;
	
	std::cout<<temp2 / 2<< " Length in samples of dry audio" << std::endl; 
	
	
	if(temp2 > temp)
	{
		temp = temp2;
	}
	
	
	
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
	
	 
	auto impPointers = bufferimp.getArrayOfReadPointers();
	auto dryPointer = bufferdry.getReadPointer(0);
	
	
	//float *out = (float*)calloc(W*H, sizeof(float));
	float* impPtrFlat = (float*)calloc(temp*channels, sizeof(float));
	float* dryPtrFlat = (float*)calloc(temp*channels, sizeof(float));
	float* outPtrFlat = (float*)calloc(temp*channels, sizeof(float));
	int counter = 0;
	for(int ch = 0; ch < channels; ch++) {
		for(int samp = 0; samp < temp; samp++) {
			impPtrFlat[counter] = impPointers[ch][samp];
			dryPtrFlat[counter] = dryPointer[samp];
			 
			counter++;
		}
	}
	
	gpuRev(impPtrFlat, dryPtrFlat, temp, outPtrFlat,channels);
	counter = 0;
	auto outPointers = bufferout.getArrayOfWritePointers();
	for(int ch = 0; ch < channels; ch++) {
		for(int samp = 0; samp < temp; samp++) {
			outPointers[ch][samp] = outPtrFlat[counter];
			counter++;
		}	
	}

	
	
	juce::WavAudioFormat format;
	std::unique_ptr<juce::AudioFormatWriter> writer;
	 
	 
	juce::File outfile = juce::File(output);
	writer.reset (format.createWriterFor (new FileOutputStream (outfile),
                                      44100.0,
                                      channels,
                                      24,
                                      {},
                                      0));
      
	if (writer != nullptr)
    	writer->writeFromAudioSampleBuffer (bufferout, channels, bufferout.getNumSamples());
	
	free(impPtrFlat);
	free(dryPtrFlat);
	free(outPtrFlat);
    return 0;
}

