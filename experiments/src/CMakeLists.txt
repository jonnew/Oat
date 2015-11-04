# Targets
add_executable (oat-exp-server matserver.cpp)
target_link_libraries (oat-exp-server ${OpenCV_LIBS} ${Boost_LIBRARIES})
install (TARGETS oat-exp-server DESTINATION ../../oat/libexec COMPONENT oat-utlities)

add_executable (oat-exp-client matclient.cpp)
target_link_libraries (oat-exp-client ${OpenCV_LIBS} ${Boost_LIBRARIES})
install (TARGETS oat-exp-client DESTINATION ../../oat/libexec COMPONENT oat-utlities)
