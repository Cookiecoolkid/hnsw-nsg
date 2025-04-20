# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加库
add_library(nsg SHARED
    index_nsg.cpp
    index.cpp
    index_graph.cpp
    index_random.cpp
)

# 设置目标文件的头文件路径
target_include_directories(nsg PUBLIC 
    ${PROJECT_SOURCE_DIR}/include
    ${EFANNA2E_INCLUDE_DIR}
    ${OpenBLAS_INCLUDE_DIR}
)

# 链接库
target_link_libraries(nsg
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

#install()