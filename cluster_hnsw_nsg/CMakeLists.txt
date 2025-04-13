cmake_minimum_required(VERSION 3.10)  # 需要升级以支持C++17
project(cluster_hnsw_nsg)

# 设置C++标准为17（FAISS需要C++14或更高）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    add_definitions(-DUSE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#OpenBLAS
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIR})
set(BLAS_LIB ${OpenBLAS_LIB})

# FAISS配置
set(FAISS_ROOT_DIR "${PROJECT_SOURCE_DIR}/../efanna_graph/extern_libraries/faiss")
include_directories(
    ${FAISS_ROOT_DIR}
    ${FAISS_ROOT_DIR}/faiss
)

# 查找FAISS库
find_library(FAISS_LIB 
    NAMES faiss
    PATHS ${FAISS_ROOT_DIR}/build/faiss
    NO_DEFAULT_PATH
    REQUIRED
)

# 其他编译选项
add_definitions(-O3 -march=native -Wall -DINFO)

# 添加测试
add_subdirectory(tests)



