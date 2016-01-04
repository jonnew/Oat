# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-buffer_SOURCE
     FrameBuffer.cpp
     TokenBuffer.cpp
     main.cpp)

# Target
add_executable (oat-buffer ${oat-buffer_SOURCE})
target_link_libraries (oat-buffer ${OatCommon_LIBS})

# Installation
install (TARGETS oat-buffer DESTINATION ../../oat/libexec COMPONENT oat-processors)
