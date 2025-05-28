# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加测试程序
add_executable(cluster_knn_nsg_search cluster_knn_nsg_search.cpp)
add_executable(cluster_knn_knn_search cluster_knn_knn_search.cpp)
add_executable(cluster_hnsw_hnsw_search cluster_hnsw_hnsw_search.cpp)
add_executable(cluster_knn_hnsw_search cluster_knn_hnsw_search.cpp)

# 链接库
target_link_libraries(cluster_knn_nsg_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_knn_knn_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_hnsw_hnsw_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_knn_hnsw_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})