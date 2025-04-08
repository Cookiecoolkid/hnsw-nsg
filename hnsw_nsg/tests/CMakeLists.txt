# 添加测试程序
add_executable(test_hnsw_nsg_search test_hnsw_nsg_search.cpp)

# 链接库到测试程序
target_link_libraries(test_hnsw_nsg_search hnsw_nsg)