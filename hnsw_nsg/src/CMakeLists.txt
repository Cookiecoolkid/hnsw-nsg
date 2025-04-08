# 添加库
add_library(hnsw_nsg SHARED
    index.cpp
    index_nsg.cpp
)

# 设置目标文件的头文件路径
target_include_directories(hnsw_nsg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)