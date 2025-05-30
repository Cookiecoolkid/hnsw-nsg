cmake_minimum_required(VERSION 3.10)  # 需要升级以支持C++17
project(cluster_hnsw_nsg)

# 设置C++标准为17（FAISS需要C++14或更高）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "No OpenMP support")
endif()

#OpenBLAS
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/module)
find_package(OpenBLAS REQUIRED)
include_directories(${OpenBLAS_INCLUDE_DIR})
set(BLAS_LIB ${OpenBLAS_LIB})

# FAISS配置
set(FAISS_ROOT_DIR "${PROJECT_SOURCE_DIR}/efanna_graph/extern_libraries/faiss")
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

# efanna2e配置
set(EFANNA2E_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/efanna_graph")
include_directories(
    ${EFANNA2E_INCLUDE_DIR}/include
)

# 项目自己的include目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/diskann_utils
    ${PROJECT_SOURCE_DIR}/include/nsg
    ${PROJECT_SOURCE_DIR}/apps/include
    ${FAISS_INCLUDE_DIRS}
    ${EFANNA2E_INCLUDE_DIR}
    ${OpenBLAS_INCLUDE_DIRS}
)

# 其他编译选项
add_definitions(-O3 -march=native -Wall -DINFO)

# 添加子目录
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(apps)

# 查找必要的包
find_package(OpenMP REQUIRED)
find_package(OpenBLAS REQUIRED)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 设置库的链接目录
link_directories(${CMAKE_BINARY_DIR}/lib)
