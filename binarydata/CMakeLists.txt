juce_add_binary_data(BinaryData SOURCES
                    wav_files/imp.wav
                    wav_files/dry.wav

)
# Make linux happy
set_target_properties(BinaryData PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
