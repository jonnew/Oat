# Test creation function
function(add_oat_test name libs)
    include_directories(${TESTING_INCLUDES})
    add_executable(${name}_test ${name}_test.cpp)
    # TODO: Cannot figure out how to pass oat-types lib from Pose_test
    target_link_libraries (${name}_test ${libs} oat-types)
    add_dependencies (${name}_test ${TESTING_INCLUDES})
    add_test(${name}_test ${name}_test)
endfunction()

# Test subdirectories
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/shmemdf)
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/datatypes)
