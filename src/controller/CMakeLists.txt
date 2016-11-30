# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-control_SOURCE
     Controller.cpp
     main.cpp)

# Target
add_executable (oat-control ${oat-control_SOURCE})
target_link_libraries (oat-control
                       oat-base
                       oat-utility 

                       ${OatCommon_LIBS})
add_dependencies (oat-control cpptoml rapidjson)

# Installation
install (TARGETS oat-control DESTINATION ../../oat/libexec COMPONENT oat-utilities)

