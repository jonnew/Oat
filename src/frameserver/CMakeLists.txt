# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Required for point-grey cameras
# TODO: super fragile. Need to provide a user flag to specify flycapture dir
include_directories ("/usr/include/flycapture")
find_library(FLYCAPTURE2 flycapture)

# Create a SOURCES variable containing all required .cpp files:
set (oat-frameserve_SOURCE PGGigECam.cpp WebCam.cpp FileReader.cpp)

# Targets
add_executable (oat-frameserve ${oat-frameserve_SOURCE} main.cpp)
add_executable (oat-calibrate ${oat-frameserve_SOURCE} calibrate.cpp)
target_link_libraries (oat-frameserve shmem ${OpenCV_LIBS} ${FLYCAPTURE2} ${Boost_LIBRARIES}) 
target_link_libraries (oat-calibrate shmem ${OpenCV_LIBS} ${FLYCAPTURE2}) 

# Installation
install (TARGETS oat-frameserve DESTINATION ../../oat/libexec COMPONENT oat-processors)
install (TARGETS oat-calibrate DESTINATION ../../oat/libexec COMPONENT oat-processors)
