SET(DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
find_package(Doxygen)

ADD_CUSTOM_TARGET(apidocs
  COMMAND ${DOXYGEN} ${DOXYFILE}
  VERBATIM)

configure_file(Doxyfile.in ${DOXYFILE} @ONLY)
