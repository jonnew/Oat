# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-framecom_SOURCE
     FrameCombiner.cpp
     MaskSubtractor.cpp
     main.cpp)

# Target
add_executable (oat-framecom ${oat-framecom_SOURCE})
target_link_libraries (oat-framecom ${OatCommon_LIBS})

# Installation
install (TARGETS oat-framecom DESTINATION ../../oat/libexec COMPONENT oat-processors)
