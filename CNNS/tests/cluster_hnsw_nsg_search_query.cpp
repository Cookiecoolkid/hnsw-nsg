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
#include <atomic>
#include <mutex>

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
    unsigned k;
    std::string prefix;
};

// 加载cluster数据
bool load_cluster_data(
    int cluster_id,
    unsigned global_dim,
    float*& cluster_data,
    unsigned& points_num,
    const std::string& prefix) {
    
    std::string cluster_filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
    unsigned dim_cluster_data;
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
        return false;
    }

    cluster_data = new (std::nothrow) float[points_num * dim_cluster_data];
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
    return true;
}

// 加载ID映射
bool load_id_mapping(
    int cluster_id,
    unsigned points_num,
    std::vector<faiss::idx_t>& id_mapping,
    const std::string& prefix) {
    
    std::string mapping_filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
    std::ifstream mapping_file(mapping_filename, std::ios::binary);
    if (!mapping_file.is_open()) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Cannot open mapping file " << mapping_filename << std::endl;
        return false;
    }

    id_mapping.resize(points_num);
    mapping_file.read((char*)id_mapping.data(), points_num * sizeof(faiss::idx_t));
    if ((unsigned)mapping_file.gcount() != points_num * sizeof(faiss::idx_t)) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error reading id_mapping file " << mapping_filename << std::endl;
        mapping_file.close();
        return false;
    }
    mapping_file.close();
    return true;
}

// 加载NSG索引
bool load_nsg_index(
    int cluster_id,
    unsigned dim,
    unsigned points_num,
    efanna2e::IndexNSG*& nsg_index,
    const std::string& prefix) {
    
    nsg_index = new (std::nothrow) efanna2e::IndexNSG(dim, points_num, efanna2e::L2, nullptr);
    if (!nsg_index) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error: Failed to allocate memory for nsg_index for cluster " << cluster_id << std::endl;
        return false;
    }

    std::string nsg_filename = prefix + "/nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
    try {
        nsg_index->Load(nsg_filename.c_str());
    } catch (const std::exception& e) {
        std::cerr << "Thread " << omp_get_thread_num() << ": Error loading NSG from " << nsg_filename << ": " << e.what() << std::endl;
        delete nsg_index;
        nsg_index = nullptr;
        return false;
    }
    return true;
}

// 按需加载指定cluster的数据和NSG索引
void load_cluster_specific_data_and_nsg(
    int cluster_id,
    unsigned global_dim,
    std::map<int, float*>& cluster_data_map,
    std::map<int, std::vector<faiss::idx_t>>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix) {

    // 加载cluster数据
    float* cluster_data = nullptr;
    unsigned points_num = 0;
    if (!load_cluster_data(cluster_id, global_dim, cluster_data, points_num, prefix)) {
        return;
    }

    // 加载ID映射
    std::vector<faiss::idx_t> id_mapping;
    if (!load_id_mapping(cluster_id, points_num, id_mapping, prefix)) {
        delete[] cluster_data;
        return;
    }

    // 加载NSG索引
    efanna2e::IndexNSG* nsg_index = nullptr;
    if (!load_nsg_index(cluster_id, global_dim, points_num, nsg_index, prefix)) {
        delete[] cluster_data;
        return;
    }

    // 成功加载，存入map
    cluster_data_map[cluster_id] = cluster_data;
    id_mapping_map[cluster_id] = id_mapping;
    cluster_nsg_indices[cluster_id] = nsg_index;
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

// 合并两个优先队列
void merge_topk_queue(
    std::priority_queue<std::pair<float, unsigned>>& target_queue,
    std::priority_queue<std::pair<float, unsigned>>& source_queue,
    unsigned k) {
    
    // 将source_queue中的所有元素添加到target_queue
    while (!source_queue.empty()) {
        auto [dist, id] = source_queue.top();
        source_queue.pop();
        
        if (target_queue.size() < k) {
            target_queue.push({dist, id});
        } else if (dist < target_queue.top().first) {
            target_queue.pop();
            target_queue.push({dist, id});
        }
    }
}

// 在NSG上搜索单个cluster
void search_nsg_cluster(
    const SearchContext& ctx,
    faiss::idx_t cluster_id,
    const float* query_data,
    std::priority_queue<std::pair<float, unsigned>>& local_queue,
    float& local_max_dist,
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

    // 更新本地优先队列
    for (const auto& [global_id, dist] : local_results) {
        if (local_queue.size() < ctx.k) {
            local_queue.push({dist, global_id});
        } else if (dist < local_queue.top().first) {
            local_queue.pop();
            local_queue.push({dist, global_id});
        }
    }

    // 更新local_max_dist
    if (!local_queue.empty()) {
        local_max_dist = local_queue.top().first;
    }

    // 如果当前cluster的最小距离大于等于当前topk中的最大距离，且队列已满，则提前停止
    if (cluster_min_dist >= local_max_dist && local_queue.size() >= ctx.k) {
        query_early_stopped = true;
    }
}

// 处理搜索结果
std::vector<unsigned> process_search_results(
    std::priority_queue<std::pair<float, unsigned>>& topk_queue,
    unsigned k) {
    
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
    for (unsigned i = 0; i < k && i < sorted_results.size(); i++) {
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
        std::atomic<int> correct{0};
        std::atomic<int> total{0};
        auto start_time_search = std::chrono::high_resolution_clock::now();
        std::atomic<double> total_hnsw_search_time{0.0};

        // 设置 omp 线程数
        omp_set_num_threads(8);

        // 并行处理每个查询
        // #pragma omp parallel
        {
            // 每个线程维护自己的统计信息
            int local_correct = 0;
            int local_total = 0;
            double local_hnsw_time = 0.0;
            std::mutex queue_mutex;  // 为每个线程创建自己的mutex

            // #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < query_num; i++) {
                // 在HNSW图上搜索并获取排序后的cluster
                auto start_time_hnsw_search = std::chrono::high_resolution_clock::now();
                auto sorted_clusters = search_hnsw_and_sort_clusters(ctx, query_data.data() + i * ctx.query_dim, nprobe);
                auto end_time_hnsw_search = std::chrono::high_resolution_clock::now();
                auto hnsw_search_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_hnsw_search - start_time_hnsw_search);
                local_hnsw_time += hnsw_search_time.count();

                std::unordered_set<unsigned> ground_truth_set(ground_truth[i].begin(), ground_truth[i].end());
                std::priority_queue<std::pair<float, unsigned>> topk_queue;
                float current_min_max_dist = std::numeric_limits<float>::max();
                bool query_early_stopped = false;

                // 使用OpenMP parallel for并行处理clusters
                #pragma omp parallel for schedule(dynamic)
                for (size_t cluster_idx = 0; cluster_idx < sorted_clusters.size(); ++cluster_idx) {
                    if (query_early_stopped) continue;

                    faiss::idx_t cluster_id = sorted_clusters[cluster_idx].first;
                    
                    // 加载cluster数据
                    bool success_load = false;
                    {
                        #pragma omp critical(cluster_load)
                        {
                            if (!ctx.cluster_nsg_indices.count(cluster_id)) {
                                load_cluster_specific_data_and_nsg(
                                    cluster_id, ctx.query_dim,
                                    ctx.cluster_data_map, ctx.id_mapping_map,
                                    ctx.cluster_nsg_indices, ctx.prefix);
                            } else {
                                success_load = true;
                            }
                        }
                    }

                    if (success_load && !query_early_stopped) {
                        std::priority_queue<std::pair<float, unsigned>> local_queue;
                        float local_max_dist = current_min_max_dist;

                        search_nsg_cluster(ctx, cluster_id, 
                                         query_data.data() + i * ctx.query_dim,
                                         local_queue, local_max_dist, 
                                         query_early_stopped);

                        // 合并结果
                        {
                            std::lock_guard<std::mutex> guard(queue_mutex);
                            merge_topk_queue(topk_queue, local_queue, ctx.k);
                            current_min_max_dist = topk_queue.top().first;
                        }
                    }
                }

                // 处理搜索结果
                auto final_results = process_search_results(topk_queue, ctx.k);
                
                // 计算recall
                int query_correct_count = calculate_query_recall(final_results, ground_truth_set);
                local_correct += query_correct_count;
                local_total += ground_truth_set.size();
            }

            // 合并线程本地统计信息
            #pragma omp critical(update_stats)
            {
                correct += local_correct;
                total += local_total;
                total_hnsw_search_time.store(total_hnsw_search_time.load() + local_hnsw_time);
            }
        }

        auto end_time_search = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> search_time = end_time_search - start_time_search;
        std::cout << "Total HNSW Search Time: " << total_hnsw_search_time.load() << " seconds" << std::endl;
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
