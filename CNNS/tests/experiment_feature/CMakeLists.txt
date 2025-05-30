# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests/experiment_feature)

# 定义通用库依赖
set(COMMON_LIBS
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    ${EFANNA2E_LIB}
    nsg
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

# 添加测试程序
add_executable(cluster_knn_nsg_search cluster_knn_nsg_search.cpp)
add_executable(cluster_knn_knn_search cluster_knn_knn_search.cpp)
add_executable(cluster_hnsw_hnsw_search cluster_hnsw_hnsw_search.cpp)
add_executable(cluster_knn_hnsw_search cluster_knn_hnsw_search.cpp)
add_executable(sort_by_min_dist sort_by_min_dist.cpp)
add_executable(pipeline_sort_by_min_dist pipeline_sort_by_min_dist.cpp)

# 链接库
foreach(TARGET cluster_knn_nsg_search cluster_knn_knn_search cluster_hnsw_hnsw_search 
        cluster_knn_hnsw_search sort_by_min_dist pipeline_sort_by_min_dist)
    target_link_libraries(${TARGET} ${COMMON_LIBS})
endforeach()