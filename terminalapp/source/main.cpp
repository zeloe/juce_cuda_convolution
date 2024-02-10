#include "cuda/kernel.h"
#include <JuceHeader.h>
#include <iostream>
#include <string>


int main()
{
 
	auto* IRStream = new juce::MemoryInputStream(BinaryData::_128_wav, BinaryData::_128_wavSize,false);
	auto* DRYStream= new juce::MemoryInputStream(BinaryData::dry_wav, BinaryData::dry_wavSize, false);
	
	WavAudioFormat wavFormat;
	std::unique_ptr<AudioFormatReader> impfile (wavFormat.createReaderFor (IRStream, false));
 
	
	const unsigned int channels = impfile->numChannels;
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
	 
	juce::File path = juce::File::getSpecialLocation(juce::File::SpecialLocationType::currentExecutableFile);
	juce::String outfile = path.getFullPathName() + "_output.wav";
	DBG(outfile);

	juce::FileOutputStream stream(outfile);

	writer.reset (format.createWriterFor ((&stream),
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

