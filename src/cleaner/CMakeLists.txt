set(oat-clean_SOURCE main.cpp)

# Target
add_executable (oat-clean ${oat-clean_SOURCE})
target_link_libraries (oat-clean ${Boost_LIBRARIES} ${Thread_LIBS})

# Installation
install(TARGETS oat-clean DESTINATION ../../oat/libexec COMPONENT oat-utilities)
