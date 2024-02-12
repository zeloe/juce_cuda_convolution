 

# Add all source files to the list of sources for the executable
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS source/*.cpp source/*.h)

# Add CUDA source files to the list of sources
file(GLOB_RECURSE CU_SOURCES CONFIGURE_DEPENDS source/cuda/*.cu)

# Add all sources to the target
target_sources(${TARGET_NAME} PRIVATE ${SOURCES})
 
# Compile CUDA source files into a shared library
add_library(CU_LIB STATIC ${CU_SOURCES})
set_target_properties(CU_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CU_LIB PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)