# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加comb_test子目录
add_subdirectory(combination_test)

# 添加测试程序
add_executable(cluster_IVF_nndescent cluster_IVF_nndescent.cpp)
add_executable(nndescent_nsg nndescent_nsg.cpp)
add_executable(cluster_hnsw_nsg_search cluster_hnsw_nsg_search.cpp)
add_executable(cluster_hnsw_nsg_search_pipeline cluster_hnsw_nsg_search_pipeline.cpp)
add_executable(cluster_hnsw_nsg_search_add_load_time cluster_hnsw_nsg_search_add_load_time.cpp)
add_executable(cluster_hnsw_nsg_search_query cluster_hnsw_nsg_search_query.cpp)
add_executable(sort_by_min_dist sort_by_min_dist.cpp)
add_executable(pipeline_sort_by_min_dist pipeline_sort_by_min_dist.cpp)

# 链接库
target_link_libraries(cluster_IVF_nndescent ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(nndescent_nsg ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(cluster_hnsw_nsg_search ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(cluster_hnsw_nsg_search_pipeline ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(cluster_hnsw_nsg_search_add_load_time ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(cluster_hnsw_nsg_search_query ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(sort_by_min_dist ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

target_link_libraries(pipeline_sort_by_min_dist ${FAISS_LIB} ${OpenBLAS_LIB} ${EFANNA2E_LIB} nsg pthread m ${OpenMP_CXX_FLAGS} aux_util)

