#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>

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

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_sift_base.fvecs>" <<
        " <ncluster>" << " <HNSW-topK>" << std::endl;
        return 1;
    }
    // 1. 加载数据
    unsigned num, dim;
    std::vector<float> data = load_fvecs(argv[1], num, dim);
    int nlist = atoi(argv[2]); // 聚类的数量  
    int k_hnsw = atoi(argv[3]); // HNSW的k值

    // 2. 创建量化器
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(dim);

    // 3. 创建IVF索引
    faiss::IndexIVFFlat* index_ivf = new faiss::IndexIVFFlat(quantizer, dim, nlist, faiss::METRIC_L2);

    // 4. 训练IVF索引
    std::vector<faiss::idx_t> ids(num);
    for (size_t i = 0; i < num; ++i) {
        ids[i] = i;
    }
    index_ivf->train(num, data.data());

    std::vector<faiss::idx_t> cluster_assignments(num); // 聚类索引

    // 获取聚类索引
    index_ivf->quantizer->assign(num, data.data(), cluster_assignments.data());

    index_ivf->add_with_ids(num, data.data(), ids.data());

    // 记录每个点的聚类索引
    std::map<faiss::idx_t, faiss::idx_t> id_to_cluster;
    for (size_t i = 0; i < num; ++i) {
        id_to_cluster[ids[i]] = cluster_assignments[i];
    }

    index_ivf->make_direct_map();

    // 5. 提取质心
    std::vector<float> centroids(nlist * dim);
    for (int i = 0; i < nlist; i++) {
        index_ivf->reconstruct(i, centroids.data() + i * dim);
    }

    // 打印质心
    for (int i = 0; i < nlist; i++) {
        std::cout << "Centroid " << i << ": ";
        for (int j = 0; j < dim; j++) {
            std::cout << centroids[i * dim + j] << " ";
        }
        std::cout << std::endl;
    }

    // // 6. 获取每个数据点的分配情况
    // std::vector<int> cluster_assignments(num);
    // for (size_t i = 0; i < num; ++i) {
    //     int cluster_idx = index_ivf->quantizer->search(&data[i * dim], 1, faiss::METRIC_L2)[0];
    //     cluster_assignments[i] = cluster_idx;
    // }

    // // 7. 获取每个质心对应的数据
    // std::vector<std::vector<float>> cluster_data(nlist);
    // for (size_t i = 0; i < num; ++i) {
    //     int cluster_idx = cluster_assignments[i];
    //     cluster_data[cluster_idx].insert(cluster_data[cluster_idx].end(), data.begin() + i * dim, data.begin() + (i + 1) * dim);
    // }

    // // 打印每个质心对应的数据
    // for (int i = 0; i < nlist; i++) {
    //     std::cout << "Cluster " << i << " has " << cluster_data[i].size() / dim << " points." << std::endl;
    // }

    // 8. 在 centroids 上构建 HNSW 图
    faiss::IndexHNSWFlat* index_hnsw = new faiss::IndexHNSWFlat(dim, k_hnsw, faiss::METRIC_L2);
    index_hnsw->add(nlist, centroids.data());

    // 9. 搜索更多示例
    std::vector<float> query(dim, 0.0); // 示例查询向量
    int k = 5; // 查询返回的邻居数量
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);

    index_hnsw->search(1, query.data(), k, distances.data(), labels.data());

    std::cout << "HNSW Search Results:" << std::endl;
    for (int i = 0; i < k; i++) {
        std::cout << "Distance: " << distances[i] << ", Label: " << labels[i] << std::endl;
    }

    // 释放资源
    delete index_ivf;
    delete quantizer;
    delete index_hnsw;

    return 0;
}