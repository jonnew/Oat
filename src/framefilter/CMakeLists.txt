# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp filesj
set (oat-framefilt_SOURCE
     BackgroundSubtractor.cpp
     BackgroundSubtractorMOG.cpp
     ColorConvert.cpp
     FrameFilter.cpp
     FrameMasker.cpp
     Undistorter.cpp
     Threshold.cpp
     main.cpp)

# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt ${OatCommon_LIBS})

# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)
