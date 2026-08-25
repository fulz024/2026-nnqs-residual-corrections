cmake_minimum_required(VERSION 3.24)
project(residual_coupling_module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
if(NOT pybind11_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE _pybind11_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _pybind11_status)
  if(_pybind11_status EQUAL 0)
    set(pybind11_DIR "${_pybind11_cmakedir}" CACHE PATH "pybind11 CMake directory")
  endif()
endif()
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
  residual_coupling_module
  article_bindings.cpp
  article_hamiltonian.cpp
  FCIDUMP.cpp)

set(
  RESIDUAL_COUPLING_MODULE_OUTPUT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/build"
  CACHE PATH "Output directory for the Python extension")
set_target_properties(residual_coupling_module PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${RESIDUAL_COUPLING_MODULE_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${RESIDUAL_COUPLING_MODULE_OUTPUT_DIR}")
foreach(_config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${_config}" _config_upper)
  set_target_properties(residual_coupling_module PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${_config_upper}" "${RESIDUAL_COUPLING_MODULE_OUTPUT_DIR}"
    "RUNTIME_OUTPUT_DIRECTORY_${_config_upper}" "${RESIDUAL_COUPLING_MODULE_OUTPUT_DIR}")
endforeach()

if(MSVC)
  target_compile_options(residual_coupling_module PRIVATE /utf-8 /W4)
else()
  target_compile_options(residual_coupling_module PRIVATE -Wall -Wextra -Wpedantic)
endif()
