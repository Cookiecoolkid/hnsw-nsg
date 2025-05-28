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
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>

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
    if (argc != 10) {
        std::cerr << "Usage: " << argv[0] << " <data_file> <n_clusters> <m_centroids> <K> <L> <iter> <S> <R> <prefix>" << std::endl;
        std::cerr << "  K: number of neighbors in NNDescent graph (default: 100)" << std::endl;
        std::cerr << "  L: number of candidates in NNDescent graph (default: 100)" << std::endl;
        std::cerr << "  iter: number of iterations (default: 10)" << std::endl;
        std::cerr << "  S: size of candidate set (default: 10)" << std::endl;
        std::cerr << "  R: maximum degree of each node (default: 100)" << std::endl;
        std::cerr << "  prefix: prefix directory for output files" << std::endl;
        return 1;
    }

    // 获取prefix参数
    std::string prefix = argv[9];

    // 创建prefix目录
    std::string mkdir_cmd = "mkdir -p " + prefix;
    auto ret = system(mkdir_cmd.c_str());
    if (ret == -1) {
        std::cerr << "Error creating directory: " << prefix << std::endl;
        return 1;
    }

    // 修改所有文件路径，添加prefix
    std::string centroids_path = prefix + "/centroids.fvecs";
    std::string hnsw_index_path = prefix + "/hnsw_memory.index";
    std::string cluster_data_dir = prefix + "/cluster_data";
    std::string nndescent_dir = prefix + "/nndescent";
    std::string mapping_dir = prefix + "/mapping";

    // 创建子目录
    ret = system(("mkdir -p " + cluster_data_dir).c_str());
    if (ret == -1) {
        std::cerr << "Error creating directory: " << cluster_data_dir << std::endl;
        return 1;
    }
    ret = system(("mkdir -p " + nndescent_dir).c_str());
    if (ret == -1) {
        std::cerr << "Error creating directory: " << nndescent_dir << std::endl;
        return 1;
    }
    ret = system(("mkdir -p " + mapping_dir).c_str());
    if (ret == -1) {
        std::cerr << "Error creating directory: " << mapping_dir << std::endl;
        return 1;
    }

    // 1. 加载数据
    unsigned num, dim;
    std::vector<float> data = load_fvecs(argv[1], num, dim);
    int n_clusters = atoi(argv[2]);
    int m = atoi(argv[3]);  // 每个cluster额外选择的点数

    // 设置NNDescent参数
    int k_nndescent = atoi(argv[4]);
    int l_nndescent = atoi(argv[5]);
    int iter = atoi(argv[6]);
    int s = atoi(argv[7]);
    int r = atoi(argv[8]);

    // 设置默认值
    if (k_nndescent == -1) k_nndescent = 100;
    if (l_nndescent == -1) l_nndescent = 100;
    if (iter == -1) iter = 10;
    if (s == -1) s = 10;
    if (r == -1) r = 100;

    std::cout << "NNDescent parameters:" << std::endl
              << "  K: " << k_nndescent << std::endl
              << "  L: " << l_nndescent << std::endl
              << "  iter: " << iter << std::endl
              << "  S: " << s << std::endl
              << "  R: " << r << std::endl;

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
    std::ofstream centroids_file(centroids_path, std::ios::binary);
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
        if ((int)ids_in_cluster.size() > m) {
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

    // 创建HNSW索引并保存
    faiss::IndexHNSWFlat* index_hnsw = new faiss::IndexHNSWFlat(dim, 32, faiss::METRIC_L2); // M = 32
    index_hnsw->add(n_clusters * (m + 1), centroids.data());
    faiss::write_index(index_hnsw, hnsw_index_path.c_str());
    std::cout << "HNSW index for centroids saved to " << hnsw_index_path << std::endl;
    delete index_hnsw;

    // 8. 为每个cluster构建NNDescent图
    for (const auto& pair : cluster_to_ids) {
        faiss::idx_t cluster_id = pair.first;
        const std::vector<faiss::idx_t>& ids_in_cluster = pair.second;

        // 保存ID映射
        std::string mapping_filename = mapping_dir + "/mapping_" + std::to_string(cluster_id);
        std::ofstream mapping_file(mapping_filename, std::ios::binary);
        mapping_file.write((char*)ids_in_cluster.data(), ids_in_cluster.size() * sizeof(faiss::idx_t));
        mapping_file.close();

        // 提取cluster数据
        float* cluster_data = new float[ids_in_cluster.size() * dim * sizeof(float)];
        for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
            memcpy(cluster_data + i * dim,
                   data.data() + ids_in_cluster[i] * dim,
                   dim * sizeof(float));
        }

        // 保存cluster数据到fvecs文件
        std::string cluster_filename = cluster_data_dir + "/cluster_" + std::to_string(cluster_id) + ".fvecs";
        std::ofstream cluster_file(cluster_filename, std::ios::binary);
        for (size_t i = 0; i < ids_in_cluster.size(); ++i) {
            cluster_file.write((char*)&dim, sizeof(dim));
            cluster_file.write((char*)(cluster_data + i * dim), dim * sizeof(float));
        }
        cluster_file.close();

        // 构建NNDescent图
        std::cout << "Building NNDescent graph for cluster " << cluster_id 
                  << " with " << ids_in_cluster.size() << " points" << std::endl;

        efanna2e::IndexRandom init_index(dim, ids_in_cluster.size());
        efanna2e::IndexGraph index(dim, ids_in_cluster.size(), efanna2e::L2, (efanna2e::Index*)(&init_index));

        efanna2e::Parameters paras;
        paras.Set<unsigned>("K", k_nndescent);
        paras.Set<unsigned>("L", l_nndescent);
        paras.Set<unsigned>("iter", iter);
        paras.Set<unsigned>("S", s);
        paras.Set<unsigned>("R", r);

        std::cout << "NNDescent parameters for cluster " << cluster_id << ":" << std::endl
                  << "  K: " << k_nndescent << std::endl
                  << "  L: " << l_nndescent << std::endl
                  << "  iter: " << iter << std::endl
                  << "  S: " << s << std::endl
                  << "  R: " << r << std::endl;

        try {
            index.Build(ids_in_cluster.size(), cluster_data, paras);
            std::cout << "Successfully built NNDescent graph for cluster " << cluster_id << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error building NNDescent graph for cluster " << cluster_id << ": " << e.what() << std::endl;
            delete[] cluster_data;
            continue;
        }

        // 保存图
        std::string graph_filename = nndescent_dir + "/nndescent_" + std::to_string(cluster_id) + ".graph";
        try {
            index.Save(graph_filename.c_str());
            std::cout << "Successfully saved graph to " << graph_filename << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error saving graph for cluster " << cluster_id << ": " << e.what() << std::endl;
            delete[] cluster_data;
            continue;
        }

        delete[] cluster_data;
    }

    // 清理资源
    delete index_ivf;
    delete quantizer;

    return 0;
}


