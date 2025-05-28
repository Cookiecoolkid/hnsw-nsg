set(CMAKE_CXX_STANDARD 17)

add_library(aux_util STATIC
    aux_util.cpp
)

target_include_directories(aux_util PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(aux_util
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)
