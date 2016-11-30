# Include the directory itself as a path to include directories
set (CMAKE_INCLUDE_CURRENT_DIR ON)

# Create a SOURCE variable containing all required .cpp files:
set (oat-calibrate_SOURCE
     Calibrator.cpp
     Saver.cpp
     UsagePrinter.cpp
     PathChanger.cpp
     CameraCalibrator.cpp
     HomographyGenerator.cpp)

# Targets
add_executable (oat-calibrate ${oat-calibrate_SOURCE} main.cpp)
target_link_libraries (oat-calibrate
                       oat-utility
                       ${OatCommon_LIBS})
add_dependencies (oat-calibrate cpptoml)

# Installation
install (TARGETS oat-calibrate DESTINATION ../../oat/libexec COMPONENT oat-utlities)
