# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-view_SOURCE
     PoseViewer.cpp
     FrameViewer.cpp
     Viewer.cpp
     main.cpp)

# Target
add_executable (oat-view ${oat-view_SOURCE})
target_link_libraries (oat-view
                       oat-base
                       oat-utility
                       ${OatCommon_LIBS})
add_dependencies (oat-view cpptoml rapidjson)

# Installation
install (TARGETS oat-view DESTINATION ../../oat/libexec COMPONENT oat-processors)
