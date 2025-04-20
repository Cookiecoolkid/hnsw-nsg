#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <iostream>
#include <fstream>  
#include <vector>
#include <string>
#include <map>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <efanna2e/util.h>
#include <filesystem>
#include <random>

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

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <n_clusters> <m_centroids>" << std::endl;
        return 1;
    }

    // 1. 加载数据
    unsigned num, dim;
    std::vector<float> data = load_fvecs(argv[1], num, dim);
    int n_clusters = atoi(argv[2]);
    int m = atoi(argv[3]);  // 每个cluster额外选择的点数

    // 2. 创建量化器
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim);

    // 3. 创建IVF索引
    faiss::IndexIVFFlat* index_ivf = new faiss::IndexIVFFlat(quantizer, dim, n_clusters, faiss::METRIC_L2);

    // 4. 训练IVF索引
    std::vector<faiss::idx_t> ids(num);
    for (size_t i = 0; i < num; ++i) {
        ids[i] = i;
    }
    index_ivf->train(num, data.data());
    index_ivf->add_with_ids(num, data.data(), ids.data());

    // 5. 获取聚类分配
    std::vector<faiss::idx_t> cluster_assignments(num);
    index_ivf->quantizer->assign(num, data.data(), cluster_assignments.data());

    // 6. 构建cluster到点id的映射
    std::map<faiss::idx_t, std::vector<faiss::idx_t>> cluster_to_ids;
    for (size_t i = 0; i < num; ++i) {
        cluster_to_ids[cluster_assignments[i]].push_back(i);
    }

    index_ivf->make_direct_map();

    // 7. 提取质心并保存
    std::vector<float> centroids((n_clusters * (m + 1)) * dim);  // 每个cluster有m+1个点（质心+m个随机点）
    std::random_device rd;
    std::mt19937 gen(rd());

    // 保存质心文件头信息
    std::ofstream centroids_file("centroids.fvecs", std::ios::binary);
    centroids_file.write((char*)&n_clusters, sizeof(n_clusters));
    centroids_file.write((char*)&m, sizeof(m));
    centroids_file.write((char*)&dim, sizeof(dim));

    for (int i = 0; i < n_clusters; i++) {
        // 保存质心
        index_ivf->reconstruct(i, centroids.data() + i * (m + 1) * dim);
        centroids_file.write((char*)&dim, sizeof(dim));
        centroids_file.write((char*)(centroids.data() + i * (m + 1) * dim), dim * sizeof(float));

        // 随机选择m个点
        const auto& ids_in_cluster = cluster_to_ids[i];
        if (ids_in_cluster.size() > m) {
            std::vector<size_t> indices(ids_in_cluster.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::shuffle(indices.begin(), indices.end(), gen);
            
            for (int j = 0; j < m; j++) {
                size_t idx = indices[j];
                memcpy(centroids.data() + (i * (m + 1) + j + 1) * dim,
                       data.data() + ids_in_cluster[idx] * dim,
                       dim * sizeof(float));
                centroids_file.write((char*)&dim, sizeof(dim));
                centroids_file.write((char*)(centroids.data() + (i * (m + 1) + j + 1) * dim), dim * sizeof(float));
            }
        } else {
            // 如果cluster中的点数不足m，则重复使用已有点
            for (int j = 0; j < m; j++) {
                size_t idx = j % ids_in_cluster.size();
                memcpy(centroids.data() + (i * (m + 1) + j + 1) * dim,
                       data.data() + ids_in_cluster[idx] * dim,
                       dim * sizeof(float));
                centroids_file.write((char*)&dim, sizeof(dim));
                centroids_file.write((char*)(centroids.data() + (i * (m + 1) + j + 1) * dim), dim * sizeof(float));
            }
        }
    }
    centroids_file.close();

    // 创建目录
    system("mkdir -p cluster_data");
    system("mkdir -p nndescent");
    system("mkdir -p nsg_mapping");  // 创建映射文件目录

    // 8. 为每个cluster构建NNDescent图
    for (const auto& pair : cluster_to_ids) {
        faiss::idx_t cluster_id = pair.first;
        const std::vector<faiss::idx_t>& ids_in_cluster = pair.second;

        // 保存ID映射
        std::string mapping_filename = "nsg_mapping/nsg_mapping_" + std::to_string(cluster_id);
        std::ofstream mapping_file(mapping_filename, std::ios::binary);
        mapping_file.write((char*)ids_in_cluster.data(), ids_in_cluster.size() * sizeof(faiss::idx_t));
        mapping_file.close();

        // 提取cluster数据
        std::vector<float> cluster_data;
        cluster_data.reserve(ids_in_cluster.size() * dim);
        cluster_data.resize(ids_in_cluster.size() * dim);

        for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
            memcpy(cluster_data.data() + i * dim,
                   data.data() + ids_in_cluster[i] * dim,
                   dim * sizeof(float));
        }

        // 保存cluster数据到fvecs文件
        std::string cluster_filename = "cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
        std::ofstream cluster_file(cluster_filename, std::ios::binary);
        for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
            cluster_file.write((char*)&dim, sizeof(dim));
            cluster_file.write((char*)(cluster_data.data() + i * dim), dim * sizeof(float));
        }
        cluster_file.close();

        // 构建NNDescent图
        efanna2e::IndexRandom init_index(dim, ids_in_cluster.size());
        efanna2e::IndexGraph index(dim, ids_in_cluster.size(), efanna2e::L2, (efanna2e::Index*)(&init_index));

        efanna2e::Parameters paras;
        int k_nndescent = 100;
        paras.Set<unsigned>("K", k_nndescent);
        paras.Set<unsigned>("L", 100);
        paras.Set<unsigned>("iter", 10);
        paras.Set<unsigned>("S", 10);
        paras.Set<unsigned>("R", 100);

        index.Build(ids_in_cluster.size(), cluster_data.data(), paras);

        // 保存图
        std::string graph_filename = "nndescent/nndescent_" + std::to_string(cluster_id) + ".graph";
        index.Save(graph_filename.c_str());
    }

    // 清理资源
    delete index_ivf;
    delete quantizer;

    return 0;
}


