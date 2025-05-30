# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源文件
set(SOURCES
    index_build.cpp
    index_search.cpp
)

# 创建库目标
add_library(cluster_hnsw_nsg STATIC ${SOURCES})

# 设置库的输出目录
set_target_properties(cluster_hnsw_nsg PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# 添加头文件目录
target_include_directories(cluster_hnsw_nsg
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/efanna_graph/include
        ${PROJECT_SOURCE_DIR}/include/diskann_utils
        ${PROJECT_SOURCE_DIR}/include/nsg
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/faiss
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/faiss/faiss
        ${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/OpenBLAS/include
        ${OpenBLAS_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
target_link_libraries(cluster_hnsw_nsg
    PUBLIC
        ${FAISS_LIB}
        ${OpenBLAS_LIB}
        ${EFANNA2E_LIB}
        nsg
        pthread
        m
        ${OpenMP_CXX_FLAGS}
)

# 添加子目录
add_subdirectory(utils)
add_subdirectory(nsg)