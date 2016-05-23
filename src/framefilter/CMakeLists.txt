# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files
set (oat-framefilt_SOURCE
     FrameFilter.cpp
     BackgroundSubtractor.cpp
     BackgroundSubtractorMOG.cpp
     FrameMasker.cpp
     Undistorter.cpp
     main.cpp)


# Target
add_executable (oat-framefilt ${oat-framefilt_SOURCE})
target_link_libraries (oat-framefilt ${OatCommon_LIBS})

if (USE_CUDA_KERNELS)
    #cuda_compile(ffk FrameFilterKernels.cu)
    cuda_add_library(ffk-lib FrameFilterKernels.cu FrameFilterKernels.h)
    target_link_libraries(oat-framefilt ffk-lib oat-cudahelper)
endif ()

# Installation
install (TARGETS oat-framefilt DESTINATION ../../oat/libexec COMPONENT oat-processors)
