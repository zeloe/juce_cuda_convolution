#Add all source files 
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS  source/*.cpp source/*.h $  source/*.cu)
# Make the folder structure visible in the IDE# Add all sources to target
target_sources(${TARGET_NAME} PRIVATE ${SOURCES})