# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
# Create a variable called helloworld_SOURCES containing all .cpp files:
set(oat-clean_SOURCE main.cpp)

# Target
add_executable (oat-clean ${oat-clean_SOURCE})
target_link_libraries (oat-clean ${OatCommon_LIBS})

# Installation
install(TARGETS oat-clean DESTINATION ../../oat/libexec COMPONENT oat-utilities)
