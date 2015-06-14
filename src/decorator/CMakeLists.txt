#include_directories(${OAT_SOURCE_DIR}/cleaner)
#link_directories(${OAT_SOURCE_DIR}/cleaner)

add_executable (oat-decorate Decorator.cpp main.cpp )
target_link_libraries (oat-decorate shmem ${OpenCV_LIBS} ${Boost_LIBRARIES}) 
	

