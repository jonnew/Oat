# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp filesj
set (oat-framefilt_SOURCE
     FrameFilter.cpp
     BackgroundSubtractor.cpp
     BackgroundSubtractorMOG.cpp
     ColorConvert.cpp
     FrameMasker.cpp
     Undistorter.cpp
     Threshold.cpp
     main.cpp)

# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt
                       oat-base
                       oat-utility
                       ${OatCommon_LIBS})
add_dependencies (oat-framefilt cpptoml)

# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)
