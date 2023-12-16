cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(TestCUDA LANGUAGES VERSION 0.0.1 CXX CUDA)

# Add JUCE as a subdirectory
add_subdirectory(modules/JUCE)



# Add your source files excluding the CUDA kernel file
juce_add_console_app(TestCUDA PRODUCT_NAME "Test CUDA")
target_sources(TestCUDA
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/source/cuda/kernel.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/source/cuda/kernel.h


)
target_compile_definitions(TestCUDA
    PRIVATE
        # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
        JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_console_app` call
        JUCE_USE_CURL=0)    # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_console_app` call
# Set C++ standard
set_property(TARGET TestCUDA PROPERTY CUDA_STANDARD 17)

# Link libraries and CUDA toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(TestCUDA
    PRIVATE
        juce::juce_core
        juce::juce_audio_utils
    	juce::juce_dsp
        CUDA::cudart
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
# Generate the JuceHeader.h file
juce_generate_juce_header(TestCUDA)
set_property(TARGET TestCUDA PROPERTY CUDA_ARCHITECTURES OFF)
