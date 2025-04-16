#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexNSG.h>
#include <faiss/IndexNNDescent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <algorithm>

// 加载fvecs文件
std::vector<float> load_fvecs(const std::string& filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    // 读取维度
    in.read((char*)&dim, 4);
    
    // 计算数据点数量
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    num = fsize / ((dim + 1) * 4);
    
    // 分配内存
    std::vector<float> data(num * dim);
    
    // 读取数据
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur); // 跳过维度头
        in.read((char*)(data.data() + i * dim), dim * sizeof(float));
    }
    in.close();
    
    return data;
}

std::vector<std::vector<unsigned>> loadGT(const char* filename) {
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    if (!in) {
        throw std::runtime_error("Cannot open file: " + std::string(filename));
    }

    std::vector<std::vector<unsigned>> results;
    while (in) {
        unsigned GK;
        in.read((char*)&GK, sizeof(unsigned));
        if (!in) break;

        std::vector<unsigned> result(GK);
        in.read((char*)result.data(), GK * sizeof(unsigned));
        if (!in) break;

        results.push_back(result);
    }

    in.close();
    return results;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <path_to_base_data>" << " <path_to_query_data>" << " <path_to_ground_truth> " << " <ncluster>" << " <HNSW-topK>" << std::endl;
        return 1;
    }
    // 1. 加载数据
    unsigned data_num, data_dim;
    unsigned query_num, query_dim;
    std::vector<float> data = load_fvecs(argv[1], data_num, data_dim);
    std::vector<float> query = load_fvecs(argv[2], query_num, query_dim);
    std::vector<std::vector<unsigned>> answers = loadGT(argv[3]);
    int nlist = atoi(argv[4]); // 聚类的数量  
    int k_hnsw = std::max(atoi(argv[5]), (int)(nlist * 0.3)); // HNSW的k值
    int k = 100; // 返回的近邻数
    int M = 32; // HNSW的邻居数 (第 0 层的邻居数)

    // 2. 创建量化器
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(data_dim);

    // 3. 创建IVF索引
    faiss::IndexIVFFlat* index_ivf = new faiss::IndexIVFFlat(quantizer, data_dim, nlist, faiss::METRIC_L2);

    // 4. 训练IVF索引
    std::vector<faiss::idx_t> ids(data_num);
    for (size_t i = 0; i < data_num; ++i) {
        ids[i] = i;
    }
    index_ivf->train(data_num, data.data());

    std::vector<faiss::idx_t> cluster_assignments(data_num); // 聚类索引

    // 获取聚类索引
    index_ivf->quantizer->assign(data_num, data.data(), cluster_assignments.data());

    index_ivf->add_with_ids(data_num, data.data(), ids.data());

    // 构建 cluster 到点 id 的映射
    std::map<faiss::idx_t, std::vector<faiss::idx_t>> cluster_to_ids;
    for (size_t i = 0; i < data_num; ++i) {
        faiss::idx_t cluster = cluster_assignments[i];
        faiss::idx_t id = ids[i];
        cluster_to_ids[cluster].push_back(id);
    }

    index_ivf->make_direct_map();

    // 5. 提取质心
    std::vector<float> centroids(nlist * data_dim);
    for (int i = 0; i < nlist; i++) {
        index_ivf->reconstruct(i, centroids.data() + i * data_dim);
    }

    // 构建质心到 cluster 的映射
    // std::map<faiss::idx_t, faiss::idx_t> centroid_to_cluster;

    // 6. 在 centroids 上构建 HNSW 图
    auto start_time_hnsw = std::chrono::high_resolution_clock::now();
    faiss::IndexHNSWFlat* index_hnsw = new faiss::IndexHNSWFlat(data_dim, M, faiss::METRIC_L2);
    index_hnsw->add(nlist, centroids.data());
    auto end_time_hnsw = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> hnsw_build_time = end_time_hnsw - start_time_hnsw;
    std::cout << "HNSW Build Time: " << hnsw_build_time.count() << " seconds" << std::endl;

    auto start_time_nsg_build = std::chrono::high_resolution_clock::now();
    // 7. 在每个 cluster 上构建 NSG 图
    // TIP：最重要的一点需要保存 NSG 点 id 到全局点 id 的映射， 否则搜索之后难以对应 id 计算 recall
    std::vector<std::vector<faiss::idx_t>> nsg_ids_to_global_ids(nlist);
    std::vector<faiss::IndexNSGFlat*> cluster_nsg_indices;
    for (const auto& pair : cluster_to_ids) {
        faiss::idx_t cluster = pair.first;
        const std::vector<faiss::idx_t>& ids_in_cluster = pair.second;

        // 创建 NSG 索引
        faiss::IndexNSGFlat* nsg_index = new faiss::IndexNSGFlat(data_dim, 32, faiss::METRIC_L2);

        nsg_index->build_type = 1;       // 使用NNDescent构建
        nsg_index->nndescent_S = 10;     // 图更新时的候选数
        nsg_index->nndescent_R = 60;     // 邻居扩展数
        nsg_index->nndescent_L = 100;    // 搜索列表大小
        nsg_index->GK = 50;              // 构建时的近邻数

        // 提取该 cluster 中的点
        std::vector<float> cluster_data(ids_in_cluster.size() * data_dim);
        for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
            faiss::idx_t id = ids_in_cluster[i];
            nsg_ids_to_global_ids[cluster].push_back(id); // 保存点 id 映射
            std::copy(data.begin() + id * data_dim, data.begin() + (id + 1) * data_dim, cluster_data.begin() + i * data_dim);
        }

        if (!nsg_index->is_trained) {
            nsg_index->train(ids_in_cluster.size(), cluster_data.data());
        }

        // 添加点到 NSG 索引
        auto start_time_nsg_add = std::chrono::high_resolution_clock::now();
        nsg_index->add(ids_in_cluster.size(), cluster_data.data());
        auto end_time_nsg_add = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> nsg_add_time = end_time_nsg_add - start_time_nsg_add;
        // std::cout << "Cluster " << cluster << " NSG Add Time: " << nsg_add_time.count() << " seconds" << std::endl;

        // 保存 NSG 索引
        cluster_nsg_indices.push_back(nsg_index);

        // 打印 NSG 图的大小
        std::cout << "Cluster " << cluster << " NSG graph size: " << nsg_index->ntotal << std::endl;
    }

    auto end_time_nsg_build = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> nsg_build_time = end_time_nsg_build - start_time_nsg_build;
    std::cout << "NSG Build Time: " << nsg_build_time.count() << " seconds" << std::endl;

    // 在 NSG 上搜索得到对应的点
    int correct = 0;
    int total = 0;

    auto start_time_search = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < query_num; i++) {
        // 在 HNSW 上搜索得到 k_hnsw 个 centroids
        std::vector<float> query_distances(k_hnsw);
        std::vector<faiss::idx_t> query_labels(k_hnsw);

        index_hnsw->search(1, query.data() + i * query_dim, k_hnsw, query_distances.data(), query_labels.data());

        // 存储所有 NSG 搜索功能的点
        std::priority_queue<std::pair<float, faiss::idx_t>, std::vector<std::pair<float, faiss::idx_t>>, 
                            std::greater<std::pair<float, faiss::idx_t>>> heap;

        // 在 NSG 上搜索得到对应的点
        for (int j = 0; j < k_hnsw; j++) {
            faiss::idx_t centroid_id = query_labels[j];
            if (centroid_id >= cluster_nsg_indices.size()) {
                continue;
            }
            faiss::IndexNSGFlat* nsg_index = cluster_nsg_indices[centroid_id];
            assert(nsg_index->is_trained);
            assert(nsg_index->is_built);
            std::vector<float> nsg_distances(k);
            std::vector<faiss::idx_t> nsg_labels(k);

            nsg_index->search(1, query.data() + i * query_dim, k, nsg_distances.data(), nsg_labels.data());

            // 将结果添加到堆中
            // 这里 label 需要添加 global id 映射
            for (int m = 0; m < k; m++) {
                heap.push({nsg_distances[m], nsg_ids_to_global_ids[centroid_id][nsg_labels[m]]});
            }
        }

        // 从堆中选择出最终的 k 个最近点
        std::vector<std::pair<float, faiss::idx_t>> final_results;
        for (int m = 0; m < k && !heap.empty(); m++) {
            final_results.push_back(heap.top());
            heap.pop();
        }
        // std::cout << "final_results size: " << final_results.size() << std::endl;

        // 计算 recall rate
        std::unordered_set<unsigned> gt_set;
        int count = 0;
        for (unsigned id : answers[i]) {
            gt_set.insert(id);
            // std::cout << "count: " << count++ << " id: " << id << std::endl;
        }

        for (auto& pair : final_results) {
            // std::cout << "final_results: " << pair.first << " " << pair.second << std::endl;
            faiss::idx_t id = pair.second;
            if (gt_set.find(id) != gt_set.end()) {
                // std::cout << "Found in ground truth: " << id << std::endl;
                correct++;
            }
        }
        total += gt_set.size();
    }
    auto end_time_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_time = end_time_search - start_time_search;
    std::cout << "Search Time: " << search_time.count() << " seconds" << std::endl;
    std::cout << "corret: " << correct << " total: " << total << std::endl;
    std::cout << "Recall Rate: " << static_cast<double>(correct) / total << std::endl;

    // 释放资源
    delete index_ivf;
    delete quantizer;
    delete index_hnsw;
    for (auto* index : cluster_nsg_indices) {
        delete index;
    }

    return 0;
}