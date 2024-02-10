cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(CUDATemplate LANGUAGES VERSION 0.0.1 CXX CUDA)


# IDEs:  Enable grouping of source files into folders in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# IDEs:  Create a folder in the IDE with the JUCE Module code.
option(JUCE_ENABLE_MODULE_SOURCE_GROUPS "Show all module sources in IDE projects" ON)




include(FetchContent)

# Keep dependencies outside of the "Build" directory.
# This allows to do a clean build of the project without re-downloading or
# rebuilding the dependencies.
set(FETCHCONTENT_BASE_DIR "${PROJECT_SOURCE_DIR}/Libs" CACHE PATH "External dependencies path." FORCE)

FetchContent_Declare(juce
    GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
    GIT_TAG        ${LIB_JUCE_TAG}
    GIT_SHALLOW    TRUE
    GIT_CONFIG     advice.detachedHead=false    # Disable detached HEAD warning for fetching a specific tag
    SOURCE_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE"
    SUBBUILD_DIR   "${FETCHCONTENT_BASE_DIR}/JUCE-Subbuild"
    BINARY_DIR     "${FETCHCONTENT_BASE_DIR}/JUCE-Build")

FetchContent_MakeAvailable(juce)



set (TARGET_NAME ${PROJECT_NAME})
 

set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")
juce_add_console_app(${TARGET_NAME}
    PRODUCT_NAME "CudaConv")
# Add your source files excluding the CUDA kernel file
add_subdirectory(terminalapp)
juce_generate_juce_header(${TARGET_NAME})
target_compile_definitions(${TARGET_NAME}
    PRIVATE
        # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
        JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_console_app` call
        JUCE_USE_CURL=0)    # If you remove this, add `NEEDS_CURL TRUE` to the `juce_add_console_app` call
# Set C++ standard
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_STANDARD 17)

# Link libraries and CUDA toolkit
find_package(CUDAToolkit REQUIRED)
target_link_libraries(${TARGET_NAME}
    PRIVATE
        juce::juce_core
        juce::juce_audio_utils
    	juce::juce_dsp
        CUDA::cudart
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_ARCHITECTURES OFF)

