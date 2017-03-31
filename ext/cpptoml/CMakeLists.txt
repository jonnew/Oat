#cmake_minimum_required(VERSION 2.8)
project(cpptoml_builder CXX)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
  cpptoml
  PREFIX ${CMAKE_BINARY_DIR}/cpptoml
  GIT_REPOSITORY https://github.com/skystrife/cpptoml.git
  TIMEOUT 10
  UPDATE_COMMAND "" #${GIT_EXECUTABLE} pull
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  LOG_DOWNLOAD ON
)

# Specify include dir
ExternalProject_Get_Property(cpptoml SOURCE_DIR)
set(CPPTOML_INCLUDE_DIR ${SOURCE_DIR}/include CACHE INTERNAL "Path to include folder for cpptoml")
