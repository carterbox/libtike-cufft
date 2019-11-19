# Must set CUDA host compiler before enabling CUDA else it will use whatever it
# finds first. Alternatively, set the environment variable CUDAHOSTCXX
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

enable_language(CUDA)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(ptycho)

# Compile a wrapper using the desired method
if(${EXTENSION_WRAPPER} STREQUAL swig)
  message("-- Using SWIG as the Python extension wrapper.")
  add_subdirectory(swig)
elseif(${EXTENSION_WRAPPER} STREQUAL pybind11)
  message("-- Using pybind11 as the Python extension wrapper.")
  add_subdirectory(pybind11)
else()
  message(
    FATAL_ERROR
    "${EXTENSION_WRAPPER} is not a valid Python extension wrapper."
  )
endif()
