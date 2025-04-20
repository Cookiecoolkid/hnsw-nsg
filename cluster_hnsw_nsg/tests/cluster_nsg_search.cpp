#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <index_nsg.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <chrono>
#include <queue>
#include <algorithm>
#include <dirent.h>
#include <regex>
#include <unordered_map>

// 加载fvecs文件
std::vector<float> load_fvecs(const std::string& filename, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&dim, 4);
    
    in.seekg(0, std::ios::end);
    size_t fsize = in.tellg();
    num = fsize / ((dim + 1) * 4);
    
    std::vector<float> data(num * dim);
    
    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, std::ios::cur);
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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <query_file> <ground_truth_file> <k>" << std::endl;
        return 1;
    }

    // 1. 加载查询数据和ground truth
    unsigned query_num, query_dim;
    std::vector<float> query_data = load_fvecs(argv[1], query_num, query_dim);
    std::vector<std::vector<unsigned>> ground_truth = loadGT(argv[2]);
    int k_HNSW = atoi(argv[3]);

    int k = 100; // 最终返回的近邻数
    int k_per_cluster = k;  //std::max(1, 100 / k_HNSW);

    std::cout << "query_num: " << query_num << " query_dim: " << query_dim << " k_HNSW: " << k_HNSW << " k: " << k << " k_per_cluster: " << k_per_cluster << std::endl;

    // 2. 加载质心数据
    unsigned centroids_num, centroids_dim;
    std::vector<float> centroids = load_fvecs("centroids.fvecs", centroids_num, centroids_dim);
    assert(centroids_dim == query_dim);

    std::cout << "centroids_num: " << centroids_num << " centroids_dim: " << centroids_dim << std::endl;

    // 3. 在质心上构建HNSW
    int M = 32; // HNSW参数
    faiss::IndexHNSWFlat* index_hnsw = new faiss::IndexHNSWFlat(centroids_dim, M, faiss::METRIC_L2);
    index_hnsw->add(centroids_num, centroids.data());

    std::cout << "index_hnsw->ntotal: " << index_hnsw->ntotal << std::endl;

    // 4. 加载所有NSG图
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, float*> cluster_data_map;  // 存储每个cluster的数据
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;  // 存储每个cluster的ID映射
    DIR* dir;
    struct dirent* ent;
    std::regex pattern("nsg_(\\d+)\\.nsg");
    std::smatch matches;

    if ((dir = opendir("nsg_graph")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (std::regex_match(filename, matches, pattern)) {
                int cluster_id = std::stoi(matches[1]);
                
                // 加载对应的cluster数据
                std::string cluster_filename = "cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";

                unsigned points_num, dim;
                std::ifstream in(cluster_filename, std::ios::binary);
                in.read((char*)&dim, 4);
                in.seekg(0, std::ios::end);
                std::ios::pos_type ss = in.tellg();
                size_t fsize = (size_t)ss;
                points_num = (unsigned)(fsize / (dim + 1) / 4);

                float* cluster_data = new float[(size_t)points_num * (size_t)dim];
                in.seekg(0, std::ios::beg);
                for (size_t i = 0; i < points_num; i++) {
                    in.seekg(4, std::ios::cur);
                    in.read((char*)(cluster_data + i * dim), dim * 4);
                }
                in.close();

                // 加载ID映射
                std::string mapping_filename = "nsg_mapping/nsg_mapping_" + std::to_string(cluster_id);
                std::ifstream mapping_file(mapping_filename, std::ios::binary);
                std::vector<faiss::idx_t> id_mapping(points_num);
                mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
                mapping_file.close();
                id_mapping_map[cluster_id] = id_mapping;

                // 创建并加载NSG索引
                efanna2e::IndexNSG* nsg_index = new efanna2e::IndexNSG(dim, points_num, efanna2e::L2, nullptr);
                nsg_index->Load(("nsg_graph/" + filename).c_str());
                cluster_nsg_indices[cluster_id] = nsg_index;
                cluster_data_map[cluster_id] = cluster_data;
            }
        }
        closedir(dir);
    }

    std::cout << "cluster_nsg_indices.size(): " << cluster_nsg_indices.size() << std::endl;

    // 5. 搜索过程
    int correct = 0;
    int total = 0;
    auto start_time_search = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < query_num; i++) {
        // 在HNSW上搜索得到k_HNSW个质心
        std::vector<float> query_distances(k_HNSW);
        std::vector<faiss::idx_t> query_labels(k_HNSW);
        index_hnsw->search(1, query_data.data() + i * query_dim, k_HNSW, 
                          query_distances.data(), query_labels.data());

        // 输出选中的cluster ID
        std::cout << "Query " << i << " selected cluster ids: ";
        for (int j = 0; j < k_HNSW; ++j) {
            std::cout << query_labels[j] << " ";
        }
        std::cout << std::endl;

        // 输出ground truth
        std::cout << "Query " << i << " ground truth: ";
        for (auto gt : ground_truth[i]) {
            std::cout << gt << " ";
        }
        std::cout << std::endl;

        // 检查ground truth分布在哪些cluster
        std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
        std::cout << "Ground truth distribution in clusters:" << std::endl;
        for (auto& [cluster_id, mapping] : id_mapping_map) {
            for (size_t local_id = 0; local_id < mapping.size(); local_id++) {
                unsigned global_id = mapping[local_id];
                if (ground_truth_set.count(global_id)) {
                    std::cout << "GT " << global_id << " in cluster " << cluster_id 
                              << " (local_id: " << local_id << ")" << std::endl;
                }
            }
        }

        // 使用map存储全局ID到距离的映射，实现去重
        std::unordered_map<unsigned, float> global_id_to_dist;

        // 在NSG上搜索得到对应的点
        for (int j = 0; j < k_HNSW; j++) {
            int cluster_id = query_labels[j];
            if (cluster_nsg_indices.find(cluster_id) == cluster_nsg_indices.end()) {
                std::cout << "Warning: cluster " << cluster_id << " not found in NSG indices" << std::endl;
                continue;
            }

            efanna2e::IndexNSG* nsg_index = cluster_nsg_indices[cluster_id];
            float* cluster_data = cluster_data_map[cluster_id];
            efanna2e::Parameters paras;
            paras.Set<unsigned>("L_search", k_per_cluster);
            paras.Set<unsigned>("P_search", k_per_cluster);

            std::vector<unsigned> tmp(k_per_cluster);
            nsg_index->Search(query_data.data() + i * query_dim, 
                            cluster_data, k_per_cluster, paras, tmp.data());

            // 将结果添加到map中，使用全局ID并去重
            for (int m = 0; m < k_per_cluster; m++) {
                unsigned local_id = tmp[m];
                if (local_id >= id_mapping_map[cluster_id].size()) {
                    std::cerr << "Error: local_id " << local_id << " out of range for cluster " << cluster_id 
                              << " (size: " << id_mapping_map[cluster_id].size() << ")" << std::endl;
                    continue;
                }
                unsigned global_id = id_mapping_map[cluster_id][local_id];
                
                // 计算实际距离
                float dist = 0;
                for (int d = 0; d < query_dim; d++) {
                    float diff = query_data[i * query_dim + d] - cluster_data[local_id * query_dim + d];
                    dist += diff * diff;
                }

                // 更新或添加距离
                if (global_id_to_dist.count(global_id)) {
                    global_id_to_dist[global_id] = std::min(global_id_to_dist[global_id], dist);
                } else {
                    global_id_to_dist[global_id] = dist;
                }
            }
        }

        // 将map转换为vector并按距离排序
        std::vector<std::pair<float, unsigned>> sorted_results;
        for (auto& [global_id, dist] : global_id_to_dist) {
            sorted_results.push_back({dist, global_id});
        }
        std::sort(sorted_results.begin(), sorted_results.end());

        // 选择前k个结果
        std::vector<unsigned> final_results;
        for (int m = 0; m < k && m < sorted_results.size(); m++) {
            final_results.push_back(sorted_results[m].second);
        }

        // 输出预测结果
        std::cout << "Query " << i << " predicted neighbors: ";
        for (auto pred : final_results) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;

        // 计算recall rate
        int query_correct = 0;
        for (unsigned id : final_results) {
            if (ground_truth_set.count(id)) {
                correct++;
                query_correct++;
            }
        }
        total += ground_truth_set.size();
        
        // 输出每个查询的recall
        std::cout << "Query " << i << " recall: " << static_cast<double>(query_correct) / ground_truth_set.size() << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    auto end_time_search = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> search_time = end_time_search - start_time_search;
    std::cout << "Search Time: " << search_time.count() << " seconds" << std::endl;
    std::cout << "correct: " << correct << " total: " << total << std::endl;
    std::cout << "Recall Rate: " << static_cast<double>(correct) / total << std::endl;

    // 清理资源
    delete index_hnsw;
    for (auto& pair : cluster_nsg_indices) {
        delete pair.second;
    }
    for (auto& pair : cluster_data_map) {
        delete[] pair.second;  // 释放cluster数据
    }

    return 0;
}
