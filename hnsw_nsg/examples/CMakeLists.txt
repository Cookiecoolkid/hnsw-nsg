# 添加可执行文件
add_executable(example_hnsw_nsg_search example_hnsw_nsg_search.cpp)

# 链接库到可执行文件
target_link_libraries(example_hnsw_nsg_search hnsw_nsg)