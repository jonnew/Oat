cmake_minimum_required (VERSION 2.8)
project (memtest)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

set (BOOST_ROOT /opt/boost_1_57_0 )
set (CPP_PTHREAD_LINK_FLAG "-lpthread")
set (CPP_RT_LINK_FLAG "-lrt")

set (CMAKE_SHARED_LINKER_FLAGS "-lpthread, -lrt")

find_package (Boost 1.57.0 EXACT REQUIRED)
find_package (OpenCV REQUIRED)

# Executables and linking
#add_executable (synctest condvar.cpp)
#target_link_libraries (synctest ${Boost_LIBRARIES} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})

#add_executable (memtest main.cpp)
#target_link_libraries (memtest ${Boost_LIBRARIES} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})

add_executable (matserver matserve.cpp)
target_link_libraries (matserver ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})

add_executable (matclient matclient.cpp)
target_link_libraries (matclient ${OpenCV_LIBS} ${Boost_LIBRARIES} ${CPP_PTHREAD_LINK_FLAG} ${CPP_RT_LINK_FLAG})
