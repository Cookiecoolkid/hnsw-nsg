#include "index_build.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <algorithm>
#include <sys/mman.h>
#include <fcntl.h>

namespace CNNS {

// 模板类实现
template<typename T>
bool IndexBuilder<T>::build(const std::string& data_file) {
    // 创建必要的目录
    std::filesystem::create_directories(prefix_);
    std::filesystem::create_directories(prefix_ + "/cluster_data");
    std::filesystem::create_directories(prefix_ + "/nndescent");
    std::filesystem::create_directories(prefix_ + "/nsg_graph");
    std::filesystem::create_directories(prefix_ + "/mapping");

    // TODO: 加载数据
    std::vector<T> data;
    unsigned dim = 0;
    // load_data(data_file, data, dim);

    // 构建IVF索引
    if (!buildIVFIndex(data, dim)) {
        std::cerr << "Failed to build IVF index" << std::endl;
        return false;
    }

    // 获取聚类分配
    std::vector<faiss::idx_t> cluster_assignments(data.size() / dim);
    index_ivf_->quantizer->assign(data.size() / dim, data.data(), cluster_assignments.data());

    // 构建cluster到点id的映射
    std::map<faiss::idx_t, std::vector<faiss::idx_t>> cluster_to_ids;
    for (size_t i = 0; i < cluster_assignments.size(); ++i) {
        cluster_to_ids[cluster_assignments[i]].push_back(i);
    }

    // 构建NNDescent图
    if (!buildNNDescentGraph(data, cluster_to_ids, dim)) {
        std::cerr << "Failed to build NNDescent graph" << std::endl;
        return false;
    }

    // 构建NSG图
    if (!buildNSGGraph(cluster_to_ids, dim)) {
        std::cerr << "Failed to build NSG graph" << std::endl;
        return false;
    }

    return true;
}

template<typename T>
bool IndexBuilder<T>::buildIVFIndex(const std::vector<T>& data, unsigned dim) {
    try {
        // 创建量化器
        quantizer_ = std::make_unique<faiss::IndexFlatL2>(dim);

        // 创建IVF索引
        index_ivf_ = std::make_unique<faiss::IndexIVFFlat>(quantizer_.get(), dim, n_clusters_, faiss::METRIC_L2);

        // 训练IVF索引
        std::vector<faiss::idx_t> ids(data.size() / dim);
        std::iota(ids.begin(), ids.end(), 0);
        index_ivf_->train(data.size() / dim, data.data());
        index_ivf_->add_with_ids(data.size() / dim, data.data(), ids.data());

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building IVF index: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNNDescentGraph(const std::vector<T>& data,
                                        const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                        unsigned dim) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            // 提取cluster数据
            std::vector<T> cluster_data(ids_in_cluster.size() * dim);
            for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
                std::copy_n(data.begin() + ids_in_cluster[i] * dim,
                           dim,
                           cluster_data.begin() + i * dim);
            }

            // 构建NNDescent图
            efanna2e::IndexRandom init_index(dim, ids_in_cluster.size());
            efanna2e::IndexGraph index(dim, ids_in_cluster.size(), efanna2e::L2, &init_index);

            efanna2e::Parameters paras;
            paras.Set<unsigned>("K", k_nndescent_);
            paras.Set<unsigned>("L", l_nndescent_);
            paras.Set<unsigned>("iter", iter_);
            paras.Set<unsigned>("S", s_);
            paras.Set<unsigned>("R", r_);

            index.Build(ids_in_cluster.size(), cluster_data.data(), paras);

            // 保存NNDescent图
            if (!saveNNDescentGraph(index, cluster_id)) {
                std::cerr << "Failed to save NNDescent graph for cluster " << cluster_id << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building NNDescent graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::buildNSGGraph(const std::map<faiss::idx_t, std::vector<faiss::idx_t>>& cluster_to_ids,
                                  unsigned dim) {
    try {
        for (const auto& [cluster_id, ids_in_cluster] : cluster_to_ids) {
            // 构建NSG
            efanna2e::IndexNSG index(dim, ids_in_cluster.size(), efanna2e::L2, nullptr);
            efanna2e::Parameters paras;
            paras.Set<unsigned>("L", L_nsg_);
            paras.Set<unsigned>("R", R_nsg_);
            paras.Set<unsigned>("C", C_nsg_);
            paras.Set<std::string>("nn_graph_path", 
                prefix_ + "/nndescent/nndescent_" + std::to_string(cluster_id) + ".graph");

            // TODO: 加载cluster数据
            std::vector<T> cluster_data;
            // load_cluster_data(cluster_id, cluster_data);

            index.Build(ids_in_cluster.size(), cluster_data.data(), paras);

            // 保存NSG
            if (!saveNSG(index, cluster_id)) {
                std::cerr << "Failed to save NSG for cluster " << cluster_id << std::endl;
                return false;
            }
        }
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error building NSG graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveCentroidsAndHNSW(const std::vector<T>& centroids, unsigned dim) {
    try {
        // 保存质心文件
        std::string centroids_path = prefix_ + "/centroids.fvecs";
        std::ofstream centroids_file(centroids_path, std::ios::binary);
        if (!centroids_file.is_open()) {
            std::cerr << "Failed to open centroids file for writing" << std::endl;
            return false;
        }

        centroids_file.write((char*)&n_clusters_, sizeof(n_clusters_));
        centroids_file.write((char*)&m_centroids_, sizeof(m_centroids_));
        centroids_file.write((char*)&dim, sizeof(dim));

        for (int i = 0; i < n_clusters_; i++) {
            centroids_file.write((char*)&dim, sizeof(dim));
            centroids_file.write((char*)(centroids.data() + i * (m_centroids_ + 1) * dim), 
                               dim * sizeof(T));
        }
        centroids_file.close();

        // 创建并保存HNSW索引
        index_hnsw_ = std::make_unique<faiss::IndexHNSWFlat>(dim, 32, faiss::METRIC_L2);
        index_hnsw_->add(n_clusters_ * (m_centroids_ + 1), centroids.data());
        faiss::write_index(index_hnsw_.get(), (prefix_ + "/hnsw_memory.index").c_str());

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving centroids and HNSW: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveNNDescentGraph(efanna2e::IndexGraph& index, 
                                       faiss::idx_t cluster_id) {
    try {
        std::string graph_filename = prefix_ + "/nndescent/nndescent_" + 
                                   std::to_string(cluster_id) + ".graph";
        index.Save(graph_filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NNDescent graph: " << e.what() << std::endl;
        return false;
    }
}

template<typename T>
bool IndexBuilder<T>::saveNSG(efanna2e::IndexNSG& index,
                            faiss::idx_t cluster_id) {
    try {
        std::string nsg_filename = prefix_ + "/nsg_graph/nsg_" + 
                                 std::to_string(cluster_id) + ".nsg";
        index.Save(nsg_filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving NSG: " << e.what() << std::endl;
        return false;
    }
}



// 显式实例化模板类
template class IndexBuilder<float>;

} // namespace CNNS
