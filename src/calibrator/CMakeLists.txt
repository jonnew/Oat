# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-calibrate_SOURCE
     Calibrator.cpp
     Saver.cpp
     UsagePrinter.cpp
     PathChanger.cpp
     HomographyGenerator.cpp)
     
# CameraCalibrator.cpp)

# Targets
add_executable (oat-calibrate ${oat-calibrate_SOURCE} main.cpp)
target_link_libraries (oat-calibrate shmem ${OpenCV_LIBS} ${Boost_LIBRARIES})
install (TARGETS oat-calibrate DESTINATION ../../oat/libexec COMPONENT oat-utlities)
