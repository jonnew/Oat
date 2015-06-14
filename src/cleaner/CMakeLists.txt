include_directories(${OAT_SOURCE_DIR}/cleaner)
link_directories(${OAT_SOURCE_DIR}/cleaner)

add_executable (oat-clean main.cpp)
target_link_libraries (oat-clean ${OpenCV_LIBS} ${Boost_LIBRARIES})

