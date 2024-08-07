cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(CudaConv LANGUAGES VERSION 0.0.1  CXX CUDA)


# IDEs:  Enable grouping of source files into folders in IDEs.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# IDEs:  Create a folder in the IDE with the JUCE Module code.
option(JUCE_ENABLE_MODULE_SOURCE_GROUPS "Show all module sources in IDE projects" ON)


set(LIB_JUCE_TAG "8.0.1")

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
 
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-10")
endif()


juce_add_console_app(${TARGET_NAME}
    PRODUCT_NAME "CudaConv")
#find_package(CUDAToolkit REQUIRED)
#set_property(TARGET ${TARGET_NAME} PROPERTY CUDA_STANDARD 17)
add_subdirectory(terminalapp)
add_subdirectory(binarydata)
enable_language(CUDA)
SET(CMAKE_CUDA_STANDARD 11)
target_compile_definitions(${TARGET_NAME}
    PRIVATE
        # JUCE_WEB_BROWSER and JUCE_USE_CURL would be on by default, but you might not need them.
        JUCE_WEB_BROWSER=0  # If you remove this, add `NEEDS_WEB_BROWSER TRUE` to the `juce_add_console_app` call
        JUCE_USE_CURL=0)     
 
target_link_libraries(${TARGET_NAME}
    PRIVATE
        BinaryData
        juce::juce_core
        juce::juce_audio_utils
    	juce::juce_dsp
        juce::juce_audio_processors
        CU_LIB
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_warning_flags
)
juce_generate_juce_header(${TARGET_NAME})