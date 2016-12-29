# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-record_SOURCE
     Format.cpp
     FrameWriter.cpp
     PositionWriter.cpp
     Writer.cpp
     #RecordControl.cpp
     Recorder.cpp
     main.cpp)

# Target
add_executable (oat-record ${oat-record_SOURCE})
target_link_libraries (oat-record
                       oat-utility
                       oat-base
                       datatypes
                       zmq
                       ${OatCommon_LIBS})
add_dependencies (oat-record cpptoml rapidjson)

# Installation
install (TARGETS oat-record DESTINATION ../../oat/libexec COMPONENT oat-processors)
