# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 测试程序1
add_executable(test_build test_build.cpp)
target_link_libraries(test_build 
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    ${EFANNA2E_LIB}
    nsg
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

# test_faissIVF需要链接更多FAISS组件
add_executable(test_faissIVF test_faissIVF.cpp)
target_link_libraries(test_faissIVF
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    ${EFANNA2E_LIB}
    nsg
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

# 添加测试程序
add_executable(cluster_IVF_nndescent cluster_IVF_nndescent.cpp)
add_executable(nndescent_nsg nndescent_nsg.cpp)
add_executable(cluster_knn_nsg_search cluster_knn_nsg_search.cpp)
add_executable(cluster_hnsw_nsg_search cluster_hnsw_nsg_search.cpp)
add_executable(cluster_knn_knn_search cluster_knn_knn_search.cpp)
add_executable(cluster_hnsw_hnsw_search cluster_hnsw_hnsw_search.cpp)
add_executable(cluster_knn_hnsw_search cluster_knn_hnsw_search.cpp)
add_executable(cluster_hnsw_nsg_search_pipeline cluster_hnsw_nsg_search_pipeline.cpp)
add_executable(cluster_hnsw_nsg_search_add_load_time cluster_hnsw_nsg_search_add_load_time.cpp)

# 链接库
target_link_libraries(cluster_IVF_nndescent ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(nndescent_nsg ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_knn_nsg_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_knn_hnsw_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_hnsw_nsg_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_hnsw_hnsw_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_knn_knn_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_hnsw_nsg_search_pipeline ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})

target_link_libraries(cluster_hnsw_nsg_search_add_load_time ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS})