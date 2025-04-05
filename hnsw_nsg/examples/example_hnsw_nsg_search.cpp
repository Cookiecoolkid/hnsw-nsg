#include "../include/index_hnsw_nsg.h"
#include <iostream>
#include <random>

int main() {
    int dim = 16;               // 数据维度
    int max_elements = 10000;   // 最大元素数量
    int M = 16;                 // 层间连接数
    int ef_construction = 200;  // 构建时的搜索范围

    // 初始化空间和索引
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW_NSG<float>* alg_hnsw_nsg = 
        new hnswlib::HierarchicalNSW_NSG<float>(&space, max_elements, M, ef_construction);

    // 生成随机数据
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // 添加数据并构建索引
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw_nsg->addPoint(data + i * dim, i);
    }
    alg_hnsw_nsg->buildLayer0(); // 显式构建NSG层

    // 查询测试
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto result = alg_hnsw_nsg->searchKnn(data + i * dim, 1);
        if (!result.empty() && result.top().second == i) correct++;
    }
    std::cout << "Recall: " << correct / max_elements << "\n";

    // 序列化索引
    std::string index_path = "hnsw_nsg.bin";
    alg_hnsw_nsg->saveIndex(index_path);
    delete alg_hnsw_nsg;

    // 反序列化验证
    alg_hnsw_nsg = new hnswlib::HierarchicalNSW_NSG<float>(&space, max_elements);
    alg_hnsw_nsg->loadIndex(index_path, &space);
    
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        auto result = alg_hnsw_nsg->searchKnn(data + i * dim, 1);
        if (!result.empty() && result.top().second == i) correct++;
    }
    std::cout << "Recall after reload: " << correct / max_elements << "\n";

    // 清理
    delete[] data;
    delete alg_hnsw_nsg;
    return 0;
}