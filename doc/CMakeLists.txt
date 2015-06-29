#set (DOXYFILE ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

# Add build target to use Doxygen to generate API documentation
find_package(Doxygen)
if (DOXYGEN_FOUND)

	add_custom_target (apidocs command ${DOXYGEN} ${DOXYFILE} VERBATIM)

	configure_file (
		${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
		${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY
		)

	add_custom_target (
		doc
		${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM
	)
elseif (not DOXYGEN_FOUND)

	message(
		AUTHOR_WARNING 
		"Doxygen is needed to build the documentation. 
		`make doc` will not work."
		)

endif (DOXYGEN_FOUND)

