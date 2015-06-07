cmake_minimum_required (VERSION 2.8)
project (Cleaner)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 

set (BOOST_ROOT /opt/boost_1_57_0 )
find_package (Boost REQUIRED system thread program_options)
link_directories (${Boost_LIBRARY_DIR})

find_package (OpenCV REQUIRED)
add_executable (oat-clean main.cpp)
target_link_libraries (oat-clean ${OpenCV_LIBS} ${Boost_LIBRARIES})

