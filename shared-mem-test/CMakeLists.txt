cmake_minimum_required (VERSION 2.8)
project (memtest)

set (BOOST_ROOT /opt/boost_1_57_0 )
set (CPP_PTHREAD_LINK_FLAG "-lpthread")
set (CPP_RT_LINK_FLAG "-lrt")
#SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG}" )
#SET( CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG}" )

set (CMAKE_SHARED_LINKER_FLAGS "-lpthread, -lrt")

find_package (Boost 1.57.0 EXACT REQUIRED)

add_executable (synctest condvar.cpp)
add_executable (memtest main.cpp)
target_link_libraries (memtest ${Boost_LIBRARIES} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})
