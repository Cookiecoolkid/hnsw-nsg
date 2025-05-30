#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
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
#include <omp.h>

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

// 新函数：按需加载指定cluster的数据和NSG索引
bool load_cluster_specific_data_and_nsg(
    int cluster_id,
    unsigned global_dim, // 全局维度，用于NSG索引初始化和数据读取
    std::map<int, float*>& cluster_data_map,
    std::map<int, std::vector<faiss::idx_t>>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices) {

    // 检查是否已加载 (这部分检查在critical区外进行一次，避免不必要的critical区进入)
    // if (cluster_nsg_indices.count(cluster_id) && cluster_data_map.count(cluster_id) && id_mapping_map.count(cluster_id)) {
    //     return true; // 如果所有相关数据都已存在，则认为已加载 (实际场景中，这个检查可能在omp critical内部更安全)
    // }
    // 这个函数的核心逻辑将在 omp critical 内部被调用，以确保线程安全地检查和加载。

    // 加载cluster数据
    std::string cluster_filename = "cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
    unsigned points_num, dim_cluster_data; // cluster内点维度应与global_dim一致
    std::ifstream in_cluster_data(cluster_filename, std::ios::binary);
    if (!in_cluster_data.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open cluster file " << cluster_filename << std::endl;
        return false;
    }
    in_cluster_data.read((char*)&dim_cluster_data, 4);
    if (dim_cluster_data != global_dim) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Dimension mismatch in cluster file " << cluster_filename 
                  << ". Expected " << global_dim << ", got " << dim_cluster_data << std::endl;
        in_cluster_data.close();
        return false;
    }
    in_cluster_data.seekg(0, std::ios::end);
    size_t fsize_cluster_data = in_cluster_data.tellg();
    points_num = fsize_cluster_data / ((dim_cluster_data + 1) * 4);

    if (points_num == 0) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cluster " << cluster_id << " has 0 points in " << cluster_filename << std::endl;
        in_cluster_data.close();
        return false; // 不能处理空cluster
    }

    float* cluster_data = new (std::nothrow) float[points_num * dim_cluster_data];
    if (!cluster_data) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for cluster_data for cluster " << cluster_id << std::endl;
        in_cluster_data.close();
        return false;
    }
    in_cluster_data.seekg(0, std::ios::beg);
    for (size_t i = 0; i < points_num; i++) {
        in_cluster_data.seekg(4, std::ios::cur); // Skip dimension for each vector
        in_cluster_data.read((char*)(cluster_data + i * dim_cluster_data), dim_cluster_data * sizeof(float));
    }
    in_cluster_data.close();

    // 加载ID映射
    std::string mapping_filename = "mapping/mapping_" + std::to_string(cluster_id);
    std::ifstream mapping_file(mapping_filename, std::ios::binary);
    if (!mapping_file.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open mapping file " << mapping_filename << std::endl;
        delete[] cluster_data;
        return false;
    }
    std::vector<faiss::idx_t> id_mapping(points_num);
    mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
    if ((unsigned)mapping_file.gcount() != points_num * sizeof(faiss::idx_t)) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error reading id_mapping file " << mapping_filename << std::endl;
        delete[] cluster_data;
        mapping_file.close();
        return false;
    }
    mapping_file.close();

    // 创建并加载NSG索引
    efanna2e::IndexNSG* nsg_index = new (std::nothrow) efanna2e::IndexNSG(dim_cluster_data, points_num, efanna2e::L2, nullptr);
    if (!nsg_index) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for nsg_index for cluster " << cluster_id << std::endl;
        delete[] cluster_data;
        return false;
    }
    std::string nsg_filename = "nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
    try {
        nsg_index->Load(nsg_filename.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error loading NSG from " << nsg_filename << ": " << e.what() << std::endl;
        delete nsg_index;
        delete[] cluster_data;
        return false;
    }

    // 成功加载，存入map
    cluster_data_map[cluster_id] = cluster_data;
    id_mapping_map[cluster_id] = id_mapping;
    cluster_nsg_indices[cluster_id] = nsg_index;
    
    // std::cout << "Thread " << omp_get_thread_num() << ": Successfully loaded data and NSG for cluster " << cluster_id << std::endl;
    return true;
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <path_to_ground_truth> <nprobe> <search_K> <search_L>" << std::endl;
        std::cerr << "  nprobe: number of clusters to search (default: 100)" << std::endl;
        std::cerr << "  search_K: number of neighbors to search in NSG (default: 100)" << std::endl;
        std::cerr << "  search_L: number of candidates in NSG search (default: 100)" << std::endl;
        return 1;
    }

    // 1. 加载查询数据和ground truth
    unsigned query_num, query_dim;
    std::vector<float> query_data = load_fvecs(argv[1], query_num, query_dim);
    std::vector<std::vector<unsigned>> ground_truth = loadGT(argv[2]);
    int nprobe = atoi(argv[3]); // 表示在质心图上搜索的邻居数（越大，最后搜索的cluster越多，召回率越高）
    int search_K = atoi(argv[4]); // NSG搜索的邻居数
    int search_L = atoi(argv[5]); // NSG搜索的候选数

    int k = 100; // 最终返回的近邻数

    std::cout << "query_num: " << query_num << " query_dim: " << query_dim 
              << " nprobe: " << nprobe 
              << " search_K: " << search_K 
              << " search_L: " << search_L
              << " k: " << k << std::endl;
    // 加载质心数据
    int n_clusters, m;
    unsigned centroids_dim;
    std::vector<float> centroids = load_centroids("centroids.fvecs", n_clusters, m, centroids_dim);
    if (centroids_dim != query_dim) {
        std::cerr << "Dimension mismatch between data and centroids" << std::endl;
        return 1;
    }

    // 加载HNSW图索引
    faiss::IndexHNSWFlat* index_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index("hnsw_memory.index"));
    if (!index_hnsw) {
        std::cerr << "Error loading HNSW index from hnsw_memory.index" << std::endl;
        return 1;
    }
    std::cout << "HNSW index loaded from hnsw_memory.index" << std::endl;

    // 初始化空的map用于存储按需加载的NSG图数据
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, float*> cluster_data_map;
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;

    // 搜索过程
    int correct = 0;
    int total = 0;
    auto start_time_search = std::chrono::high_resolution_clock::now();

    // 设置 omp 线程数
    omp_set_num_threads(8);

    #pragma omp parallel for
    for (size_t i = 0; i < query_num; i++) {
        // 在HNSW图上搜索得到nprobe个点
        std::vector<float> query_distances(nprobe);
        std::vector<faiss::idx_t> query_labels(nprobe);
        index_hnsw->search(1, query_data.data() + i * query_dim, nprobe, 
                          query_distances.data(), query_labels.data());

        // 修改排序逻辑，使用样本点到查询点的距离
        std::map<faiss::idx_t, float> cluster_min_dist; // 记录每个簇中最近样本点到查询点的距离
        std::map<faiss::idx_t, int> cluster_sample_count; // 保留样本点计数，可能后续会用到

        for (int j = 0; j < nprobe; ++j) {
            faiss::idx_t point_id = query_labels[j];
            faiss::idx_t cluster_id = point_id / (m + 1);
            cluster_sample_count[cluster_id]++;
            
            // 计算当前样本点到查询点的距离
            float dist = 0;
            for (unsigned d = 0; d < query_dim; d++) {
                float diff = query_data[i * query_dim + d] - centroids[point_id * query_dim + d];
                dist += diff * diff;
            }
            
            // 更新该簇的最小距离
            if (cluster_min_dist.count(cluster_id)) {
                cluster_min_dist[cluster_id] = std::min(cluster_min_dist[cluster_id], dist);
            } else {
                cluster_min_dist[cluster_id] = dist;
            }
        }

        // 将cluster按最小距离排序
        std::vector<std::pair<faiss::idx_t, float>> sorted_clusters;
        for (const auto& pair : cluster_min_dist) {
            sorted_clusters.push_back(pair);
        }
        std::sort(sorted_clusters.begin(), sorted_clusters.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });

        // 使用set记录已搜索过的簇
        std::unordered_set<faiss::idx_t> searched_clusters;

        // 检查ground truth分布在哪些cluster
        std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());

        // 使用map存储全局ID到距离的映射
        std::unordered_map<unsigned, float> global_id_to_dist;
        float current_min_max_dist = std::numeric_limits<float>::max();

        // 在NSG上搜索得到对应的点
        for (const auto& [cluster_id, min_dist] : sorted_clusters) {
            // 如果该簇已经搜索过，跳过
            if (searched_clusters.count(cluster_id)) {
                continue;
            }
            
            // 标记该簇为已搜索
            searched_clusters.insert(cluster_id);
            
            // 按需加载 cluster 数据和 NSG 索引，需要线程安全保护
            bool is_loaded_or_load_successful = false;
            if (cluster_nsg_indices.count(cluster_id)) {
                is_loaded_or_load_successful = true;
            } else {
                #pragma omp critical
                {
                    if (!cluster_nsg_indices.count(cluster_id)) {
                        if (load_cluster_specific_data_and_nsg(cluster_id, query_dim, 
                            cluster_data_map, id_mapping_map, cluster_nsg_indices)) {
                            is_loaded_or_load_successful = true;
                        }
                    } else {
                        is_loaded_or_load_successful = true;
                    }
                }
            }
            
            if (!is_loaded_or_load_successful) {
                // 输出警告可以考虑放在 critical 区外，或者只在 load_cluster_specific_data_and_nsg 内部处理
                // std::cout << "Thread " << omp_get_thread_num() << ": Warning: cluster " << cluster_id << " could not be loaded or found." << std::endl;
                continue; 
            }

            efanna2e::IndexNSG* nsg_index = cluster_nsg_indices[cluster_id];
            float* current_cluster_data = cluster_data_map[cluster_id]; // 使用 current_cluster_data 避免重定义
            const auto& current_id_mapping = id_mapping_map[cluster_id]; // 使用 current_id_mapping

            efanna2e::Parameters paras;
            paras.Set<unsigned>("L_search", search_L);
            paras.Set<unsigned>("P_search", search_L);
            paras.Set<unsigned>("K_search", search_K);

            std::vector<unsigned> tmp(search_K);
            nsg_index->Search(query_data.data() + i * query_dim, 
                            current_cluster_data, search_K, paras, tmp.data());

            // 计算当前cluster中最小距离
            float cluster_min_dist = std::numeric_limits<float>::max();
            for (int m_loop = 0; m_loop < search_K; m_loop++) { // 重命名循环变量避免与最外层 m 冲突
                unsigned local_id = tmp[m_loop];
                if (local_id >= current_id_mapping.size()) { // 使用 current_id_mapping
                    std::cerr << "Error: local_id " << local_id << " out of range for cluster " << cluster_id 
                              << " (size: " << current_id_mapping.size() << ")" << std::endl; // 使用 current_id_mapping
                    continue;
                }
                unsigned global_id = current_id_mapping[local_id]; // 使用 current_id_mapping
                
                // 计算实际距离
                float dist = 0;
                for (unsigned d = 0; d < query_dim; d++) {
                    float diff = query_data[i * query_dim + d] - current_cluster_data[local_id * query_dim + d]; // 使用 current_cluster_data
                    dist += diff * diff;
                }
                cluster_min_dist = std::min(cluster_min_dist, dist);

                // 更新或添加距离
                if (global_id_to_dist.count(global_id)) {
                    global_id_to_dist[global_id] = std::min(global_id_to_dist[global_id], dist);
                } else {
                    global_id_to_dist[global_id] = dist;
                }
            }

            // 如果当前cluster的最小距离大于等于之前所有cluster中最大距离的最小值，则停止搜索
            if (cluster_min_dist >= current_min_max_dist) {
                break;
            }

            // 更新当前最小最大距离
            if (!global_id_to_dist.empty()) {
                std::vector<float> distances;
                distances.reserve(global_id_to_dist.size());
                for (const auto& pair : global_id_to_dist) {
                    distances.push_back(pair.second);
                }
                std::sort(distances.begin(), distances.end());
                current_min_max_dist = std::min(current_min_max_dist, distances[std::min(k - 1, (int)distances.size() - 1)]);
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

        // 计算recall rate
        int query_correct = 0;
        for (unsigned id : final_results) {
            if (ground_truth_set.count(id)) {
                correct++;
                query_correct++;
            }
        }
        total += ground_truth_set.size();
        
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
        delete[] pair.second;
    }

    return 0;
}