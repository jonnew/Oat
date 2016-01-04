# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-record_SOURCE
     RecordControl.cpp
     Recorder.cpp
     main.cpp)

# Target
add_executable (oat-record ${oat-record_SOURCE})
target_link_libraries (oat-record
                       oatutility
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES}
                       ${Thread_LIBS}
                       zmq)

# Installation
install (TARGETS oat-record DESTINATION ../../oat/libexec COMPONENT oat-processors)
