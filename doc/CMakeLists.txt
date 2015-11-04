# Add build target to use Doxygen to generate API documentation (optional)
# Enables "make docs"
option(BUILD_DOCS "Use Doxygen to create the HTML documentation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen)
  if (NOT DOXYGEN_FOUND)
    message(FATAL_ERROR "Doxygen can't be found.")
  endif()
  configure_file(Doxyfile.in ${PROJECT_BINARY_DIR}/doc/Doxyfile @ONLY IMMEDIATE)
  add_custom_target (docs
    ${DOXYGEN_EXECUTABLE}
    ${PROJECT_BINARY_DIR}/doc/Doxyfile
    SOURCES
    ${PROJECT_BINARY_DIR}/doc/Doxyfile)
endif()
