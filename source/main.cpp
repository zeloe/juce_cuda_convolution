#include "cuda/kernel.h"
#include <JuceHeader.h>



int main()
{
	
	juce::AudioFormatManager formatManager;

	formatManager.registerBasicFormats();
	juce::File file = juce::File(" ");
	auto* impfile = formatManager.createReaderFor(file);
	auto newSource = std::make_shared<juce::AudioFormatReaderSource>(impfile, true);
	juce::AudioSampleBuffer bufferimp;
	int temp = ((int(impfile->lengthInSamples / 512) +1) * 512) * 2;
	bufferimp.setSize(1, temp);
	impfile->read(&bufferimp, 0, temp, 0, true, false);
	const float* dataimp = bufferimp.getReadPointer(0);
	 


	juce::AudioFormatManager formatManager2;

	formatManager2.registerBasicFormats();
	juce::File file2 = juce::File(" ");
	auto* dry = formatManager2.createReaderFor(file2);
	auto newSourcedry = std::make_shared<juce::AudioFormatReaderSource>(dry, true);
	juce::AudioSampleBuffer bufferdry;
	bufferdry.setSize(1, temp);
	dry->read(&bufferdry, 0, impfile->lengthInSamples, 0, true, false);
	const float* datadry = bufferdry.getReadPointer(0);
	


	int threads = 1024;

	int blocks = ((int)(temp / threads)) + 1;
	juce::AudioBuffer<float> buffer2;
	buffer2.setSize(1,temp);
	float* h_wetBuffer = buffer2.getWritePointer(0);
	//(const float* dryBuffer, const int dryBufferSize, const float* irBuffer, const int irBufferSize, float* d_wetBuffer, int blocks, int threads, int N, int Ndry)
	gpuRev(datadry, dataimp, temp, blocks, threads,h_wetBuffer);
	
	
	
	

	
	
	juce::WavAudioFormat format;
	std::unique_ptr<juce::AudioFormatWriter> writer;
	//applyGain (int channel, int startSample, int numSamples, Type gain) noexcept
	buffer2.applyGain(0,0,buffer2.getNumSamples(), 0.15);
	juce::File outfile = juce::File(" ");
	writer.reset (format.createWriterFor (new FileOutputStream (outfile),
                                      44100.0,
                                      1,
                                      24,
                                      {},
                                      0));
      
	if (writer != nullptr)
    	writer->writeFromAudioSampleBuffer (buffer2, 0, buffer2.getNumSamples());
	//gpuRev(datadry, Ndry, dataimp, N, wetPtr, blocks, threads, N,Ndry);
	//cudaDeviceSynchronize();

    return 0;
}

