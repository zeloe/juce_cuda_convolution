juce_add_binary_data(BinaryData SOURCES
                    wav_files/128.wav
                    wav_files/dry.wav

)
# Make linux happy
set_target_properties(BinaryData PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
