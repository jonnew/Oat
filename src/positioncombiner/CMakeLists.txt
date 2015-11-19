# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-posicom_SOURCE
     PositionCombiner.cpp
     MeanPosition.cpp
     main.cpp)

# Target
add_executable (oat-posicom ${oat-posicom_SOURCE})
target_link_libraries (oat-posicom shmem ${OpenCV_LIBS} ${Boost_LIBRARIES})

# Installation
install (TARGETS oat-posicom DESTINATION ../../oat/libexec COMPONENT oat-processors)
