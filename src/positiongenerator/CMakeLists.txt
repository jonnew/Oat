# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCES variable containing all required .cpp files:
set (oat-posigen_SOURCE
     PositionGenerator.cpp
     RandomAccel2D.cpp
     main.cpp)

# Target
add_executable (oat-posigen ${oat-posigen_SOURCE})
target_link_libraries (oat-posigen 
                       oat-utility
                       ${OatCommon_LIBS})
add_dependencies (oat-posigen cpptoml rapidjson)

# Installation
install (TARGETS oat-posigen DESTINATION ../../oat/libexec COMPONENT oat-processors)
