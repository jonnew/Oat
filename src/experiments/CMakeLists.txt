# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Targets
add_executable (oat-exp-server matserver.cpp)
target_link_libraries (oat-exp-server shmem ${OpenCV_LIBS} ${Boost_LIBRARIES})
install (TARGETS oat-exp-server DESTINATION ../../oat/libexec COMPONENT oat-utlities)

add_executable (oat-exp-client matclient.cpp)
target_link_libraries (oat-exp-client  shmem ${OpenCV_LIBS} ${Boost_LIBRARIES})
install (TARGETS oat-exp-client DESTINATION ../../oat/libexec COMPONENT oat-utlities)