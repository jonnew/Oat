# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-posisock_SOURCE
     PositionSocket.cpp
     UDPPositionClient.cpp
     UDPPositionServer.cpp
     main.cpp)

# Target
add_executable (oat-posisock ${oat-posisock_SOURCE})
target_link_libraries (oat-posisock
                       ${OpenCV_LIBS}
                       ${Boost_LIBRARIES}
                       ${Thread_LIBS})

# Installation
install (TARGETS oat-posisock DESTINATION ../../oat/libexec COMPONENT oat-processors)
