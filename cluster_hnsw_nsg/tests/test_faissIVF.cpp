#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

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
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_sift_base.fvecs>" << std::endl;
        return 1;
    }
    // 1. 加载数据
    unsigned num, dim;
    std::vector<float> data = load_fvecs(argv[1], num, dim);
    int d = 128; // 数据维度
    int nlist = 100; // 聚类的数量

    // 2. 创建量化器
    faiss::IndexFlatL2* quantizer = new faiss::IndexFlatL2(d);

    // 3. 创建IVF索引
    faiss::IndexIVFFlat* index_ivf = new faiss::IndexIVFFlat(quantizer, d, nlist, faiss::METRIC_L2);

    // 4. 训练IVF索引
    std::vector<faiss::idx_t> ids(num);
    for (size_t i = 0; i < num; ++i) {
        ids[i] = i;
    }
    index_ivf->train(num, data.data());
    index_ivf->add_with_ids(num, data.data(), ids.data());

    index_ivf->make_direct_map();

    // 5. 提取质心
    std::vector<float> centroids(nlist * d);
    for (int i = 0; i < nlist; i++) {
        index_ivf->reconstruct(i, centroids.data() + i * d);
    }

    // 打印质心
    for (int i = 0; i < nlist; i++) {
        std::cout << "Centroid " << i << ": ";
        for (int j = 0; j < d; j++) {
            std::cout << centroids[i * d + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放资源
    delete index_ivf;
    delete quantizer;

    return 0;
}