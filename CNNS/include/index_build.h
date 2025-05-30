#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>

#include "nsg/neighbor.h"
#include "nsg/index_nsg.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace CNNS {

template<typename T>
class IndexBuilder {
public:
    // 构造函数
    IndexBuilder(const std::string& prefix,
                int n_clusters,
                int m_centroids,
                int k_nndescent = 100,
                int l_nndescent = 100,
                int iter = 10,
                int s = 10,
                int r = 100,
                int L_nsg = 40,
                int R_nsg = 50,
                int C_nsg = 500);

    // 析构函数
    ~IndexBuilder();

    // 构建索引
    bool build(const std::string& data_file);

    // 释放资源
    void release();

private:
    // 构建IVF索引
    bool buildIVFIndex(const std::vector<T>& data, unsigned dim);

    // 构建NNDescent图
    bool buildNNDescentGraph(const std::vector<T>& data, 
                            const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                            unsigned dim);

    // 构建NSG图
    bool buildNSGGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                      unsigned dim);

    // 保存质心和HNSW索引
    bool saveCentroidsAndHNSW(const std::vector<T>& centroids, unsigned dim);

    // 保存NNDescent图
    bool saveNNDescentGraph(efanna2e::IndexGraph& index, 
                           faiss::idx_t cluster_id);

    // 保存NSG图
    bool saveNSG(efanna2e::IndexNSG& index,
                faiss::idx_t cluster_id);

    // 成员变量
    std::string prefix_;
    int n_clusters_;
    int m_centroids_;
    int k_nndescent_;
    int l_nndescent_;
    int iter_;
    int s_;
    int r_;
    int L_nsg_;
    int R_nsg_;
    int C_nsg_;

    // 索引相关指针
    std::unique_ptr<faiss::IndexIVFFlat> index_ivf_;
    std::unique_ptr<faiss::IndexFlatL2> quantizer_;
    std::unique_ptr<faiss::IndexHNSWFlat> index_hnsw_;
};
} // namespace CNNS
