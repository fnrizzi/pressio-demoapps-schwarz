# Largely copied from pressiodemoapps, to make sure we're compatible

cmake_minimum_required(VERSION 3.18.0)
project(pdaschwarz CXX)

#=====================================================================
# we need c++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++17" COMPILER_SUPPORT_CPP17)
if(NOT COMPILER_SUPPORT_CPP17)
  message(FATAL_ERROR "Compiler does not support -std=c++17. This is required.")
endif()
set(PRESSIO_ENABLE_CXX17 ON)

#=====================================================================
# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR "In-source builds are not allowed.
    Please make a new directory (e.g. build directory) and run CMake from there.")
endif()

# default to release if build type is empty
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "${CMAKE_BUILD_TYPE} is not specified, defaulting to Release.")
  set(CMAKE_BUILD_TYPE "Release")
endif()

# convert cmake build type to lower string
string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if( NOT cmake_build_type_tolower STREQUAL "debug"
    AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR "Unknown build type \"${CMAKE_BUILD_TYPE}\".
    Allowed values are Debug, Release (case-insensitive).")
endif()

#=====================================================================
# We're not going to mess with OpenMP or Python bindings for now
# Just check that pressiodemoapps is specified correctly

if(NOT PDA_SOURCE)
  message(FATAL_ERROR "Must specify pressiodemoapps source directory via PDA_SOURCE")
else()
  if(NOT EXISTS ${PDA_SOURCE})
    message(NOTICE "PDA_SOURCE: ${PDA_SOURCE}")
    message(FATAL_ERROR "PDA_SOURCE does not exist or is not an absolute path")
  endif()
endif()

if(NOT PRESSIO_SOURCE)
  message(FATAL_ERROR "Must specify pressio source directory via PRESSIO_SOURCE")
else()
  if(NOT EXISTS ${PRESSIO_SOURCE})
    message(NOTICE "PRESSIO_SOURCE: ${PRESSIO_SOURCE}")
    message(FATAL_ERROR "PRESSIO_SOURCE does not exist or is not an absolute path")
  endif()
endif()

option(ENABLE_PERF OFF)
message(NOTICE ${ENABLE_PERF})
if(${ENABLE_PERF})
  set(CMAKE_BUILD_TYPE "Release")
  add_subdirectory(perf)
else()
  enable_testing()
  add_subdirectory(tests_cpp)
endif()
