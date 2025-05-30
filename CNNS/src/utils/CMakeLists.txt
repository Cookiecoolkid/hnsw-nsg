set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(aux_util STATIC
    aux_util.cpp
)

add_library(diskann_utils STATIC
    ann_exception.cpp
    linux_aligned_file_reader.cpp
    logger.cpp
    windows_aligned_file_reader.cpp
)

target_include_directories(diskann_utils PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include/diskann_utils
)

target_link_libraries(diskann_utils 
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

target_compile_options(diskann_utils PRIVATE
    -O3
    -march=native
    -Wall
    -DINFO
    -msse4.2
    -mavx2
    -mfma
)
