# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a SOURCE variable containing all required .cpp files:
set (oat-decorate_SOURCE Decorator.cpp main.cpp)

# Target
add_executable (oat-decorate ${oat-decorate_SOURCE})
target_link_libraries (oat-decorate 
                       oatutility
                       ${OatCommon_LIBS})
add_dependencies (oat-decorate cpptoml rapidjson)

# Installation
install (TARGETS oat-decorate DESTINATION ../../oat/libexec COMPONENT oat-processors)
