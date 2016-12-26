# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-posisock_SOURCE
     PositionCout.cpp
     PositionSocket.cpp
     PositionPublisher.cpp
     PositionReplier.cpp
     UDPPositionClient.cpp
     # TODO: There are minor but debilitating issues with this component that I
     # dont have time to fix now
     #UDPPositionServer.cpp 
     main.cpp)

# Target
add_executable (oat-posisock ${oat-posisock_SOURCE})
target_link_libraries (oat-posisock 
                       oat-utility
                       oat-base
                       zmq
                       ${OatCommon_LIBS})
add_dependencies (oat-posisock cpptoml rapidjson)

# Installation
install (TARGETS oat-posisock DESTINATION ../../oat/libexec COMPONENT oat-processors)
