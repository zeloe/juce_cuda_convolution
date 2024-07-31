 

# Add all source files to the list of sources for the executable
file(GLOB SOURCES CONFIGURE_DEPENDS source/*.cpp source/*.h)

# Add CUDA source files to the list of sources
file(GLOB CU_SOURCES CONFIGURE_DEPENDS source/cuda/*.cu source/cuda/*.cuh)

# Add all sources to the target
target_sources(${TARGET_NAME} PRIVATE ${SOURCES})
 
# Compile CUDA source files into a static library
add_library(CU_LIB STATIC ${CU_SOURCES})
set_target_properties(CU_LIB PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET CU_LIB PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Ensure CU_LIB is compiled with nvcc
set_target_properties(CU_LIB PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_source_files_properties(${CU_SOURCES} PROPERTIES LANGUAGE CUDA)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
# Add a special target to clean nvcc generated files.
