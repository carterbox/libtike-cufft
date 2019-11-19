# Build the CUDA components as a regular static library
add_library(libptychofft STATIC ptychofft.cu)
target_link_libraries(libptychofft PRIVATE cudart cufft)
target_compile_features(libptychofft PUBLIC cxx_std_11)
set_target_properties(libptychofft PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
)
