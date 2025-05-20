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
        std::cerr << "  nprobe: number of clusters to search (default: 50)" << std::endl;
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
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices; // Global, shared by threads
    std::map<int, float*> cluster_data_map; // Global, shared by threads
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map; // Global, shared by threads

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

        // 统计每个cluster包含的样本点数量
        std::map<faiss::idx_t, int> cluster_sample_count;
        for (int j = 0; j < nprobe; ++j) {
            faiss::idx_t point_id = query_labels[j];
            faiss::idx_t cluster_id = point_id / (m + 1); // m is loaded from centroids.fvecs header
            cluster_sample_count[cluster_id]++;
        }

        std::vector<std::pair<faiss::idx_t, int>> sorted_clusters;
        for (const auto& pair : cluster_sample_count) {
            sorted_clusters.push_back(pair);
        }
        std::sort(sorted_clusters.begin(), sorted_clusters.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });

        std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
        std::unordered_map<unsigned, float> global_id_to_dist_for_query; // Query-local results
        float current_min_max_dist_for_query = std::numeric_limits<float>::max(); // Query-local
        std::unordered_set<faiss::idx_t> searched_clusters_for_query; // Query-local

        bool query_early_stopped = false;

        // --- 处理第一个簇 (阻塞式加载和搜索) ---
        if (!sorted_clusters.empty()) {
            faiss::idx_t first_cluster_id = sorted_clusters[0].first;
            bool first_cluster_load_successful = false;

            // 检查是否已加载，如果未加载，则阻塞式加载
            #pragma omp critical(load_and_search_cluster_logic)
            {
                if (!cluster_nsg_indices.count(first_cluster_id)) {
                    // std::cout << "Thread " << omp_get_thread_num() << ": First cluster " << first_cluster_id << " not loaded. Loading obstructively." << std::endl;
                    if (load_cluster_specific_data_and_nsg(first_cluster_id, query_dim, cluster_data_map, id_mapping_map, cluster_nsg_indices)) {
                        first_cluster_load_successful = true;
                    } else {
                        // std::cerr << "Thread " << omp_get_thread_num() << ": Failed to load critical first cluster " << first_cluster_id << std::endl;
                    }
                } else { // 已被其他线程或过程加载
                    first_cluster_load_successful = true;
                }
            } // 结束临界区

            if (first_cluster_load_successful) {
                efanna2e::IndexNSG* nsg_index = cluster_nsg_indices.at(first_cluster_id); // map::at会抛异常如果key不存在
                float* current_cluster_data = cluster_data_map.at(first_cluster_id);
                const auto& current_id_mapping = id_mapping_map.at(first_cluster_id);
                
                efanna2e::Parameters paras;
                paras.Set<unsigned>("L_search", search_L);
                paras.Set<unsigned>("P_search", search_L);
                paras.Set<unsigned>("K_search", search_K);
                std::vector<unsigned> tmp(search_K);
                nsg_index->Search(query_data.data() + i * query_dim, 
                                current_cluster_data, search_K, paras, tmp.data());

                float cluster_min_dist = std::numeric_limits<float>::max();
                for (int m_loop = 0; m_loop < search_K; m_loop++) {
                    unsigned local_id = tmp[m_loop];
                    if (local_id >= current_id_mapping.size()) continue;
                    unsigned global_id = current_id_mapping[local_id];
                    float dist = 0;
                    for (unsigned d_idx = 0; d_idx < query_dim; d_idx++) { // Renamed d to d_idx
                        float diff = query_data[i * query_dim + d_idx] - current_cluster_data[local_id * query_dim + d_idx];
                        dist += diff * diff;
                    }
                    cluster_min_dist = std::min(cluster_min_dist, dist);
                    if (global_id_to_dist_for_query.count(global_id)) {
                        global_id_to_dist_for_query[global_id] = std::min(global_id_to_dist_for_query[global_id], dist);
                    } else {
                        global_id_to_dist_for_query[global_id] = dist;
                    }
                }
                if (cluster_min_dist >= current_min_max_dist_for_query && !global_id_to_dist_for_query.empty()) {
                    query_early_stopped = true;
                }
                if (!global_id_to_dist_for_query.empty()) {
                    std::vector<float> distances_vec;
                    for(const auto& p : global_id_to_dist_for_query) distances_vec.push_back(p.second);
                    std::sort(distances_vec.begin(), distances_vec.end());
                    if (!distances_vec.empty()){
                         current_min_max_dist_for_query = std::min(current_min_max_dist_for_query, distances_vec[std::min((int)k - 1, (int)distances_vec.size() - 1)]);
                    }
                }
                searched_clusters_for_query.insert(first_cluster_id);
            }
        }
        // --- 第一个簇处理结束 ---


        // --- 处理剩余的簇 ---
        // PASS 1 (for remaining clusters): Search already loaded clusters
        // Iterate from the second cluster onwards (index 1)
        if (!query_early_stopped) { // Only proceed if query not already stopped
            for (size_t cluster_idx = 1; cluster_idx < sorted_clusters.size(); ++cluster_idx) {
                if (query_early_stopped) break;
                faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;
                // No need to check searched_clusters_for_query here for Pass 1,
                // as first cluster is handled above, and this pass only looks for *already loaded* ones.

                bool is_loaded = false;
                #pragma omp critical(map_read_check_pass1_remaining) 
                { 
                    if (cluster_nsg_indices.count(cluster_id)) {
                        is_loaded = true;
                    }
                }

                if (is_loaded) {
                    // Ensure it wasn't the first cluster if by some chance it got re-evaluated
                    // (though current logic for first cluster processing should prevent this for Pass 1)
                    if (searched_clusters_for_query.count(cluster_id)) continue;


                    efanna2e::IndexNSG* nsg_index = cluster_nsg_indices.at(cluster_id);
                    float* current_cluster_data = cluster_data_map.at(cluster_id);
                    const auto& current_id_mapping = id_mapping_map.at(cluster_id);
                    
                    efanna2e::Parameters paras;
                    paras.Set<unsigned>("L_search", search_L);
                    paras.Set<unsigned>("P_search", search_L);
                    paras.Set<unsigned>("K_search", search_K);
                    std::vector<unsigned> tmp(search_K);
                    nsg_index->Search(query_data.data() + i * query_dim, 
                                    current_cluster_data, search_K, paras, tmp.data());

                    float cluster_min_dist = std::numeric_limits<float>::max();
                    for (int m_loop = 0; m_loop < search_K; m_loop++) {
                        unsigned local_id = tmp[m_loop];
                        if (local_id >= current_id_mapping.size()) continue;
                        unsigned global_id = current_id_mapping[local_id];
                        float dist = 0;
                        for (unsigned d_idx = 0; d_idx < query_dim; d_idx++) { // Renamed d to d_idx
                            float diff = query_data[i * query_dim + d_idx] - current_cluster_data[local_id * query_dim + d_idx];
                            dist += diff * diff;
                        }
                        cluster_min_dist = std::min(cluster_min_dist, dist);
                        if (global_id_to_dist_for_query.count(global_id)) {
                            global_id_to_dist_for_query[global_id] = std::min(global_id_to_dist_for_query[global_id], dist);
                        } else {
                            global_id_to_dist_for_query[global_id] = dist;
                        }
                    }
                    if (cluster_min_dist >= current_min_max_dist_for_query && !global_id_to_dist_for_query.empty()) { 
                        query_early_stopped = true;
                    }
                    if (!global_id_to_dist_for_query.empty()) {
                        std::vector<float> distances_vec;
                        for(const auto& p : global_id_to_dist_for_query) distances_vec.push_back(p.second);
                        std::sort(distances_vec.begin(), distances_vec.end());
                        if (!distances_vec.empty()){
                             current_min_max_dist_for_query = std::min(current_min_max_dist_for_query, distances_vec[std::min((int)k - 1, (int)distances_vec.size() - 1)]);
                        }
                    }
                    searched_clusters_for_query.insert(cluster_id);
                }
            }
        }

        // PASS 2 (for remaining clusters): Load and search remaining clusters if not early-stopped
        if (!query_early_stopped) {
            // Iterate from the second cluster onwards (index 1)
            for (size_t cluster_idx = 1; cluster_idx < sorted_clusters.size(); ++cluster_idx) {
                if (query_early_stopped) break;
                faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;
                if (searched_clusters_for_query.count(cluster_id)) continue; // Already searched (either first cluster or in Pass 1 for remaining)

                bool successfully_loaded_this_pass = false;

                #pragma omp critical(load_check_and_run_pass2_remaining)
                {
                    // Double check if it got loaded by another thread between Pass 1 and Pass 2 for this query
                    if (cluster_nsg_indices.count(cluster_id)) { 
                        successfully_loaded_this_pass = true; 
                    } else {
                        // std::cout << "Thread " << omp_get_thread_num() << ": P2 for remaining, attempting to load cluster " << cluster_id << std::endl;
                        if (load_cluster_specific_data_and_nsg(cluster_id, query_dim, cluster_data_map, id_mapping_map, cluster_nsg_indices)) {
                            successfully_loaded_this_pass = true;
                        } else {
                            // std::cerr << "Thread " << omp_get_thread_num() << ": P2 for remaining, failed to load " << cluster_id << std::endl;
                        }
                    }
                }

                if (successfully_loaded_this_pass) {
                    efanna2e::IndexNSG* nsg_index = cluster_nsg_indices.at(cluster_id);
                    float* current_cluster_data = cluster_data_map.at(cluster_id);
                    const auto& current_id_mapping = id_mapping_map.at(cluster_id);

                    efanna2e::Parameters paras;
                    paras.Set<unsigned>("L_search", search_L);
                    paras.Set<unsigned>("P_search", search_L);
                    paras.Set<unsigned>("K_search", search_K);
                    std::vector<unsigned> tmp(search_K);
                    nsg_index->Search(query_data.data() + i * query_dim, 
                                    current_cluster_data, search_K, paras, tmp.data());

                    float cluster_min_dist = std::numeric_limits<float>::max();
                    for (int m_loop = 0; m_loop < search_K; m_loop++) {
                        unsigned local_id = tmp[m_loop];
                        if (local_id >= current_id_mapping.size()) continue;
                        unsigned global_id = current_id_mapping[local_id];
                        float dist = 0;
                        for (unsigned d_idx = 0; d_idx < query_dim; d_idx++) { // Renamed d to d_idx
                            float diff = query_data[i * query_dim + d_idx] - current_cluster_data[local_id * query_dim + d_idx];
                            dist += diff * diff;
                        }
                        cluster_min_dist = std::min(cluster_min_dist, dist);
                        if (global_id_to_dist_for_query.count(global_id)) {
                            global_id_to_dist_for_query[global_id] = std::min(global_id_to_dist_for_query[global_id], dist);
                        } else {
                            global_id_to_dist_for_query[global_id] = dist;
                        }
                    }
                    if (cluster_min_dist >= current_min_max_dist_for_query && !global_id_to_dist_for_query.empty()) {
                         query_early_stopped = true;
                    }
                    if (!global_id_to_dist_for_query.empty()) {
                        std::vector<float> distances_vec;
                        for(const auto& p : global_id_to_dist_for_query) distances_vec.push_back(p.second);
                        std::sort(distances_vec.begin(), distances_vec.end());
                        if(!distances_vec.empty()){
                            current_min_max_dist_for_query = std::min(current_min_max_dist_for_query, distances_vec[std::min((int)k - 1, (int)distances_vec.size() - 1)]);
                        }
                    }
                    searched_clusters_for_query.insert(cluster_id); 
                }
            }
        }
        // --- 处理剩余的簇结束 ---


        // Consolidate results for query i
        std::vector<std::pair<float, unsigned>> all_results_for_query;
        all_results_for_query.reserve(global_id_to_dist_for_query.size());
        for (auto& [g_id, dist_val] : global_id_to_dist_for_query) {
            all_results_for_query.push_back({dist_val, g_id});
        }
        std::partial_sort(all_results_for_query.begin(), 
                          all_results_for_query.begin() + std::min((int)k, (int)all_results_for_query.size()), 
                          all_results_for_query.end());
        std::vector<unsigned> final_results_for_query;
        final_results_for_query.reserve(k);
        for (int m_final = 0; m_final < k && m_final < (int)all_results_for_query.size(); m_final++) {
            final_results_for_query.push_back(all_results_for_query[m_final].second);
        }

        // Calculate recall for this query and update global correct and total counts (needs to be thread-safe)
        int query_correct_count = 0;
        for (unsigned id : final_results_for_query) {
            if (ground_truth_set.count(id)) {
                query_correct_count++;
            }
        }
        #pragma omp critical(update_recall_counts)
        {
            correct += query_correct_count;
            total += ground_truth_set.size();
        }
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
