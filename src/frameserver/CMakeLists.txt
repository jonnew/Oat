# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
if (${OAT_USE_FLYCAP})
    set (oat-frameserve_SOURCE
         PGGigECam.cpp
         WebCam.cpp
         FileReader.cpp)
else (${OAT_USE_FLYCAP})
    set (oat-frameserve_SOURCE
         WebCam.cpp
         FileReader.cpp)
endif (${OAT_USE_FLYCAP})

# Targets
add_executable (oat-frameserve ${oat-frameserve_SOURCE} main.cpp)
target_link_libraries (oat-frameserve ${OpenCV_LIBS} ${FLYCAPTURE2} ${Boost_LIBRARIES})

# Installation
install (TARGETS oat-frameserve DESTINATION ../../oat/libexec COMPONENT oat-processors)
