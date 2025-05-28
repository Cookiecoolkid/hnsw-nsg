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
#include <aux_util.h>

// 数据结构定义
struct SearchContext {
    std::map<int, efanna2e::IndexNSG*> cluster_nsg_indices;
    std::map<int, float*> cluster_data_map;
    std::map<int, std::vector<faiss::idx_t>> id_mapping_map;
    faiss::IndexHNSWFlat* index_hnsw;
    int n_clusters;
    int m;
    unsigned query_dim;
    int search_K;
    int search_L;
    int k;
    std::string prefix;
};

// 新函数：按需加载指定cluster的数据和NSG索引
bool load_cluster_specific_data_and_nsg(
    int cluster_id,
    unsigned global_dim,
    std::map<int, float*>& cluster_data_map,
    std::map<int, std::vector<faiss::idx_t>>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix) {

    // 加载cluster数据
    std::string cluster_filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
    unsigned points_num, dim_cluster_data;
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
    std::string mapping_filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
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
    std::string nsg_filename = prefix + "/nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
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
    
    return true;
}

// 初始化搜索上下文
SearchContext initialize_search_context(
    const std::string& query_data_path,
    const std::string& ground_truth_path,
    int nprobe,
    int search_K,
    int search_L,
    const std::string& prefix) {
    
    SearchContext ctx;
    ctx.search_K = search_K;
    ctx.search_L = search_L;
    ctx.k = 100;
    ctx.prefix = prefix;

    // 加载查询数据和ground truth
    unsigned query_num;
    std::vector<float> query_data = CNNS::load_fvecs(query_data_path, query_num, ctx.query_dim);
    std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(ground_truth_path.c_str());

    // 加载质心数据
    unsigned centroids_dim;
    std::vector<float> centroids = CNNS::load_centroids(prefix + "/centroids.fvecs", ctx.n_clusters, ctx.m, centroids_dim);
    if (centroids_dim != ctx.query_dim) {
        throw std::runtime_error("Dimension mismatch between data and centroids");
    }

    // 加载HNSW图索引
    ctx.index_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index((prefix + "/hnsw_memory.index").c_str()));
    if (!ctx.index_hnsw) {
        throw std::runtime_error("Error loading HNSW index from " + prefix + "/hnsw_memory.index");
    }

    return ctx;
}

// 在HNSW图上搜索并获取排序后的cluster
std::vector<std::pair<faiss::idx_t, int>> search_hnsw_and_sort_clusters(
    const SearchContext& ctx,
    const float* query_data,
    int nprobe) {
    
    std::vector<float> query_distances(nprobe);
    std::vector<faiss::idx_t> query_labels(nprobe);
    
    ctx.index_hnsw->search(1, query_data, nprobe, query_distances.data(), query_labels.data());

    // 统计每个cluster包含的样本点数量
    std::map<faiss::idx_t, int> cluster_sample_count;
    for (int j = 0; j < nprobe; ++j) {
        faiss::idx_t point_id = query_labels[j];
        faiss::idx_t cluster_id = point_id / (ctx.m + 1);
        cluster_sample_count[cluster_id]++;
    }

    // 将cluster按样本点数量排序
    std::vector<std::pair<faiss::idx_t, int>> sorted_clusters;
    for (const auto& pair : cluster_sample_count) {
        sorted_clusters.push_back(pair);
    }
    std::sort(sorted_clusters.begin(), sorted_clusters.end(),
             [](const auto& a, const auto& b) { return a.second > b.second; });

    return sorted_clusters;
}

// 在NSG上搜索单个cluster
void search_nsg_cluster(
    const SearchContext& ctx,
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& topk_queue,
    float& current_min_max_dist,
    bool& query_early_stopped) {

    efanna2e::IndexNSG* nsg_index = ctx.cluster_nsg_indices.at(cluster_id);
    float* current_cluster_data = ctx.cluster_data_map.at(cluster_id);
    const auto& current_id_mapping = ctx.id_mapping_map.at(cluster_id);
    
    efanna2e::Parameters paras;
    paras.Set<unsigned>("L_search", ctx.search_L);
    paras.Set<unsigned>("P_search", ctx.search_L);
    paras.Set<unsigned>("K_search", ctx.search_K);
    std::vector<unsigned> tmp(ctx.search_K);
    nsg_index->Search(query_data, current_cluster_data, ctx.search_K, paras, tmp.data());

    float cluster_min_dist = std::numeric_limits<float>::max();
    std::unordered_map<unsigned, float> local_results;

    // 计算当前cluster中所有点的距离
    for (int m_loop = 0; m_loop < ctx.search_K; m_loop++) {
        unsigned local_id = tmp[m_loop];
        if (local_id >= current_id_mapping.size()) continue;
        unsigned global_id = current_id_mapping[local_id];
        float dist = 0;
        for (unsigned d_idx = 0; d_idx < ctx.query_dim; d_idx++) {
            float diff = query_data[d_idx] - current_cluster_data[local_id * ctx.query_dim + d_idx];
            dist += diff * diff;
        }
        cluster_min_dist = std::min(cluster_min_dist, dist);
        local_results[global_id] = dist;
    }

    // 更新优先队列
    for (const auto& [global_id, dist] : local_results) {
        // 如果队列未满，直接添加
        if (topk_queue.size() < ctx.k) {
            topk_queue.push({dist, global_id});
        }
        // 如果队列已满且当前距离小于队列中最大距离，则替换
        else if (dist < topk_queue.top().first) {
            topk_queue.pop();
            topk_queue.push({dist, global_id});
        }
    }

    // 更新current_min_max_dist为当前队列中的最大距离
    if (!topk_queue.empty()) {
        current_min_max_dist = topk_queue.top().first;
    }

    // 如果当前cluster的最小距离大于等于当前topk中的最大距离，且队列已满，则提前停止
    if (cluster_min_dist >= current_min_max_dist && topk_queue.size() >= ctx.k) {
        query_early_stopped = true;
    }
}

// 处理搜索结果
std::vector<unsigned> process_search_results(
    std::priority_queue<std::pair<float, unsigned>>& topk_queue,
    int k) {
    
    std::vector<unsigned> final_results;
    final_results.reserve(k);
    
    // 将优先队列中的结果按距离从小到大排序
    std::vector<std::pair<float, unsigned>> sorted_results;
    sorted_results.reserve(topk_queue.size());
    while (!topk_queue.empty()) {
        sorted_results.push_back(topk_queue.top());
        topk_queue.pop();
    }
    std::sort(sorted_results.begin(), sorted_results.end());
    
    // 提取前k个结果
    for (int i = 0; i < k && i < sorted_results.size(); i++) {
        final_results.push_back(sorted_results[i].second);
    }
    
    return final_results;
}

// 计算单个查询的recall
int calculate_query_recall(
    const std::vector<unsigned>& final_results,
    const std::unordered_set<unsigned>& ground_truth_set) {
    
    int correct = 0;
    for (unsigned id : final_results) {
        if (ground_truth_set.count(id)) {
            correct++;
        }
    }
    return correct;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " <path_to_query_data> <path_to_ground_truth> <nprobe> <search_K> <search_L> <prefix>" << std::endl;
        std::cerr << "  nprobe: number of clusters to search (default: 50)" << std::endl;
        std::cerr << "  search_K: number of neighbors to search in NSG (default: 100)" << std::endl;
        std::cerr << "  search_L: number of candidates in NSG search (default: 100)" << std::endl;
        std::cerr << "  prefix: directory prefix for all data files" << std::endl;
        return 1;
    }

    try {
        // 初始化搜索上下文
        SearchContext ctx = initialize_search_context(argv[1], argv[2], atoi(argv[3]), 
                                                   atoi(argv[4]), atoi(argv[5]), argv[6]);

        // 加载查询数据
        unsigned query_num;
        std::vector<float> query_data = CNNS::load_fvecs(argv[1], query_num, ctx.query_dim);
        std::vector<std::vector<unsigned>> ground_truth = CNNS::loadGT(argv[2]);
        int nprobe = atoi(argv[3]);

        std::cout << "query_num: " << query_num << " query_dim: " << ctx.query_dim 
                  << " nprobe: " << nprobe 
                  << " search_K: " << ctx.search_K 
                  << " search_L: " << ctx.search_L
                  << " k: " << ctx.k << std::endl;

        // 搜索过程
        int correct = 0;
        int total = 0;
        auto start_time_search = std::chrono::high_resolution_clock::now();

        // 设置 omp 线程数
        omp_set_num_threads(8);

        #pragma omp parallel
        {
            #pragma omp for
            for (size_t i = 0; i < query_num; i++) {
                // 在HNSW图上搜索并获取排序后的cluster
                auto sorted_clusters = search_hnsw_and_sort_clusters(ctx, query_data.data() + i * ctx.query_dim, nprobe);

                std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
                std::priority_queue<std::pair<float, unsigned>> topk_queue;
                float current_min_max_dist_for_query = std::numeric_limits<float>::max();
                std::unordered_set<faiss::idx_t> searched_clusters_for_query;
                bool query_early_stopped = false;

                // 处理第一个簇
                if (!sorted_clusters.empty()) {
                    faiss::idx_t first_cluster_id = sorted_clusters[0].first;
                    bool first_cluster_load_successful = false;

                    #pragma omp critical(load_and_search_cluster_logic)
                    {
                        if (!ctx.cluster_nsg_indices.count(first_cluster_id)) {
                            if (load_cluster_specific_data_and_nsg(first_cluster_id, ctx.query_dim, 
                                                                ctx.cluster_data_map, ctx.id_mapping_map, 
                                                                ctx.cluster_nsg_indices, ctx.prefix)) {
                                first_cluster_load_successful = true;
                            }
                        } else {
                            first_cluster_load_successful = true;
                        }
                    }

                    if (first_cluster_load_successful) {
                        search_nsg_cluster(ctx, first_cluster_id, query_data.data() + i * ctx.query_dim,
                                         topk_queue, current_min_max_dist_for_query,
                                         query_early_stopped);
                        searched_clusters_for_query.insert(first_cluster_id);
                    }
                }

                // 处理剩余的簇
                if (!query_early_stopped) {
                    for (size_t cluster_idx = 1; cluster_idx < sorted_clusters.size(); ++cluster_idx) {
                        if (query_early_stopped) break;
                        faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;
                        if (searched_clusters_for_query.count(cluster_id)) continue;

                        bool is_loaded = false;
                        #pragma omp critical(map_read_check_pass1_remaining)
                        {
                            is_loaded = ctx.cluster_nsg_indices.count(cluster_id) > 0;
                        }

                        if (is_loaded) {
                            search_nsg_cluster(ctx, cluster_id, query_data.data() + i * ctx.query_dim,
                                             topk_queue, current_min_max_dist_for_query,
                                             query_early_stopped);
                            searched_clusters_for_query.insert(cluster_id);
                        } else {
                            bool successfully_loaded_this_pass = false;
                            #pragma omp critical(load_check_and_run_pass2_remaining)
                            {
                                if (!ctx.cluster_nsg_indices.count(cluster_id)) {
                                    if (load_cluster_specific_data_and_nsg(cluster_id, ctx.query_dim,
                                                                        ctx.cluster_data_map, ctx.id_mapping_map,
                                                                        ctx.cluster_nsg_indices, ctx.prefix)) {
                                        successfully_loaded_this_pass = true;
                                    }
                                } else {
                                    successfully_loaded_this_pass = true;
                                }
                            }
                        }
                    }
                }

                // 处理搜索结果
                auto final_results = process_search_results(topk_queue, ctx.k);
                
                // 计算recall
                int query_correct_count = calculate_query_recall(final_results, ground_truth_set);
                #pragma omp critical(update_recall_counts)
                {
                    correct += query_correct_count;
                    total += ground_truth_set.size();
                }
            }
        }

        auto end_time_search = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> search_time = end_time_search - start_time_search;
        std::cout << "Total Search Time: " << search_time.count() << " seconds" << std::endl;
        std::cout << "correct: " << correct << " total: " << total << std::endl;
        std::cout << "Recall Rate: " << static_cast<double>(correct) / total << std::endl;

        // 清理资源
        delete ctx.index_hnsw;
        for (auto& pair : ctx.cluster_nsg_indices) {
            delete pair.second;
        }
        for (auto& pair : ctx.cluster_data_map) {
            delete[] pair.second;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
