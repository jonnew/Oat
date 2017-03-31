#cmake_minimum_required(VERSION 2.8)
project(rapidjson_builder CXX)
include(ExternalProject)
find_package(Git REQUIRED)

ExternalProject_Add(
  rapidjson
  PREFIX ${CMAKE_BINARY_DIR}/rapidjson
  GIT_REPOSITORY https://github.com/miloyip/rapidjson.git
  TIMEOUT 10
  UPDATE_COMMAND "" #${GIT_EXECUTABLE} pull
  CONFIGURE_COMMAND ""
  INSTALL_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  LOG_DOWNLOAD ON
)

# Specify include dir
ExternalProject_Get_Property(rapidjson SOURCE_DIR)
set(RAPIDJSON_INCLUDE_DIR ${SOURCE_DIR}/include CACHE INTERNAL "Path to include folder for rapidjson")
