#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
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

// 加载质心
std::vector<float> load_centroids(const std::string& filename, int& n_clusters, int& m, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&n_clusters, sizeof(n_clusters));
    in.read((char*)&m, sizeof(m));
    in.read((char*)&dim, sizeof(dim));
    
    size_t total_points = n_clusters * (m + 1);
    std::vector<float> centroids(total_points * dim);
    
    for (size_t i = 0; i < total_points; ++i) {
        unsigned point_dim;
        in.read((char*)&point_dim, sizeof(point_dim));
        if (point_dim != dim) {
            std::cerr << "Dimension mismatch in centroids file" << std::endl;
            exit(1);
        }
        in.read((char*)(centroids.data() + i * dim), dim * sizeof(float));
    }
    
    in.close();
    return centroids;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <path_to_ground_truth> <nprobe>" << std::endl;
        return 1;
    }

    // 1. 加载查询数据和ground truth
    unsigned query_num, query_dim;
    std::vector<float> query_data = load_fvecs(argv[1], query_num, query_dim);
    std::vector<std::vector<unsigned>> ground_truth = loadGT(argv[2]);
    int nprobe = atoi(argv[3]); // 表示在质心图上搜索的邻居数（越大，最后搜索的cluster越多，召回率越高）

    int k = 100; // 最终返回的近邻数
    int k_per_cluster = k;  // 每个cluster返回的近邻数

    std::cout << "query_num: " << query_num << " query_dim: " << query_dim << " nprobe: " << nprobe << " k: " << k << " k_per_cluster: " << k_per_cluster << std::endl;

    // 加载质心
    int n_clusters, m;
    unsigned centroids_dim;
    std::vector<float> centroids = load_centroids("centroids.fvecs", n_clusters, m, centroids_dim);
    if (centroids_dim != query_dim) {
        std::cerr << "Dimension mismatch between data and centroids" << std::endl;
        return 1;
    }

    // 创建KNN图索引
    faiss::IndexFlatL2* index_flat = new faiss::IndexFlatL2(query_dim);
    index_flat->add(n_clusters * (m + 1), centroids.data());

    // 加载所有cluster数据
    std::map<int, float*> cluster_data_map;
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;
    DIR* dir;
    struct dirent* ent;
    std::regex pattern("cluster_(\\d+)\\.fvecs");

    if ((dir = opendir("cluster_data")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            std::smatch matches;
            if (std::regex_match(filename, matches, pattern)) {
                int cluster_id = std::stoi(matches[1]);
                
                // 加载cluster数据
                std::string cluster_filename = "cluster_data/" + filename;
                unsigned points_num, dim;
                std::ifstream in(cluster_filename, std::ios::binary);
                if (!in.is_open()) {
                    std::cerr << "Error: Cannot open cluster file " << cluster_filename << std::endl;
                    continue;
                }
                in.read((char*)&dim, 4);
                in.seekg(0, std::ios::end);
                size_t fsize = in.tellg();
                points_num = fsize / ((dim + 1) * 4);

                float* cluster_data = new float[points_num * dim * sizeof(float)];
                in.seekg(0, std::ios::beg);
                for (size_t i = 0; i < points_num; i++) {
                    in.seekg(4, std::ios::cur);
                    in.read((char*)(cluster_data + i * dim), dim * 4);
                }
                in.close();

                // 加载ID映射
                std::string mapping_filename = "mapping/mapping_" + std::to_string(cluster_id);
                std::ifstream mapping_file(mapping_filename, std::ios::binary);
                if (!mapping_file.is_open()) {
                    std::cerr << "Error: Cannot open mapping file " << mapping_filename << std::endl;
                    delete[] cluster_data;
                    continue;
                }
                std::vector<faiss::idx_t> id_mapping(points_num);
                mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
                mapping_file.close();

                cluster_data_map[cluster_id] = cluster_data;
                id_mapping_map[cluster_id] = id_mapping;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Error: Cannot open cluster_data directory" << std::endl;
        return 1;
    }

    std::cout << "Loaded " << cluster_data_map.size() << " clusters" << std::endl;

    // 搜索过程
    int correct = 0;
    int total = 0;
    auto start_time_search = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < query_num; i++) {
        // 在KNN图上搜索得到nprobe个点
        std::vector<float> query_distances(nprobe);
        std::vector<faiss::idx_t> query_labels(nprobe);
        index_flat->search(1, query_data.data() + i * query_dim, nprobe, 
                          query_distances.data(), query_labels.data());

        // 获取不同的cluster ID
        std::unordered_set<faiss::idx_t> selected_clusters;
        for (int j = 0; j < nprobe; ++j) {
            faiss::idx_t point_id = query_labels[j];
            faiss::idx_t cluster_id = point_id / (m + 1);
            selected_clusters.insert(cluster_id);
        }

        // 输出选中的cluster ID
        std::cout << "Query " << i << " selected cluster ids: ";
        for (auto cluster_id : selected_clusters) {
            std::cout << cluster_id << " ";
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

        // 使用map存储全局ID到距离的映射
        std::unordered_map<unsigned, float> global_id_to_dist;

        // 在每个选中的cluster中进行KNN搜索
        for (auto cluster_id : selected_clusters) {
            if (cluster_data_map.find(cluster_id) == cluster_data_map.end()) {
                std::cout << "Warning: cluster " << cluster_id << " not found in cluster data" << std::endl;
                continue;
            }

            float* cluster_data = cluster_data_map[cluster_id];
            const auto& id_mapping = id_mapping_map[cluster_id];
            size_t points_num = id_mapping.size();

            // 创建KNN索引
            faiss::IndexFlatL2 cluster_index(query_dim);
            cluster_index.add(points_num, cluster_data);

            // 搜索
            std::vector<float> cluster_distances(k_per_cluster);
            std::vector<faiss::idx_t> cluster_labels(k_per_cluster);
            cluster_index.search(1, query_data.data() + i * query_dim, k_per_cluster,
                               cluster_distances.data(), cluster_labels.data());

            // 将结果添加到map中
            for (int m = 0; m < k_per_cluster; m++) {
                unsigned local_id = cluster_labels[m];
                if (local_id >= id_mapping.size()) {
                    std::cerr << "Error: local_id " << local_id << " out of range for cluster " << cluster_id 
                              << " (size: " << id_mapping.size() << ")" << std::endl;
                    continue;
                }
                unsigned global_id = id_mapping[local_id];
                float dist = cluster_distances[m];

                // 更新或添加距离
                if (global_id_to_dist.count(global_id)) {
                    global_id_to_dist[global_id] = std::min(global_id_to_dist[global_id], dist);
                } else {
                    global_id_to_dist[global_id] = dist;
                }
            }
        }

        // 将map转换为vector并按距离排序
        std::vector<std::pair<float, unsigned>> all_results;
        all_results.reserve(global_id_to_dist.size());
        for (auto& [global_id, dist] : global_id_to_dist) {
            all_results.push_back({dist, global_id});
        }

        // 使用partial_sort进行部分排序
        std::partial_sort(all_results.begin(), all_results.begin() + std::min(k, (int)all_results.size()), 
                         all_results.end());

        // 选择前k个结果
        std::vector<unsigned> final_results;
        final_results.reserve(k);
        for (int m = 0; m < k && m < (int)all_results.size(); m++) {
            final_results.push_back(all_results[m].second);
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
    delete index_flat;
    for (auto& pair : cluster_data_map) {
        delete[] pair.second;
    }

    return 0;
}
