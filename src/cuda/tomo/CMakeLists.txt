# Build the CUDA components as a regular static library
add_library(libradonusfft STATIC radonusfft.cu)
target_link_libraries(libradonusfft PRIVATE cudart cufft)
target_compile_features(libradonusfft PUBLIC cxx_std_11)
set_target_properties(libradonusfft PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_VISIBILITY_PRESET "hidden"
    # CUDA_SEPARABLE_COMPILATION ON
    PREFIX ""
)
