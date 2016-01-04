# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
if (${USE_FLYCAP})
    set (oat-frameserve_SOURCE
         TestFrame.cpp
         PGGigECam.cpp
         WebCam.cpp
         FileReader.cpp)
else (${USE_FLYCAP})
    set (oat-frameserve_SOURCE
         TestFrame.cpp
         WebCam.cpp
         FileReader.cpp)
endif (${USE_FLYCAP})

# Targets
add_executable (oat-frameserve ${oat-frameserve_SOURCE} main.cpp)
target_link_libraries (oat-frameserve 
                       ${OpenCV_LIBS} 
                       ${FLYCAPTURE2} 
                       ${Boost_LIBRARIES} 
                       ${Thread_LIBS})

# Installation
install (TARGETS oat-frameserve DESTINATION ../../oat/libexec COMPONENT oat-processors)
