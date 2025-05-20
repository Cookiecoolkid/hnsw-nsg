#include <index_nsg.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <dirent.h>
#include <regex>

// 加载fvecs文件
void load_data(const std::string& filename, float*& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Open file error: " << filename << std::endl;
        exit(1);
    }
    
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim+1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <L> <R> <C>" << std::endl;
        std::cerr << "  L: number of candidates in NSG (default: 40)" << std::endl;
        std::cerr << "  R: maximum degree of each node (default: 50)" << std::endl;
        std::cerr << "  C: number of candidates in NSG (default: 500)" << std::endl;
        return 1;
    }

    unsigned L = (unsigned)atoi(argv[1]);
    unsigned R = (unsigned)atoi(argv[2]);
    unsigned C = (unsigned)atoi(argv[3]);

    // 加载所有cluster数据
    DIR* dir;
    struct dirent* ent;
    std::regex pattern("cluster_(\\d+)\\.fvecs");

    // 创建nsg_graph目录
    auto ret = system("mkdir -p nsg_graph");
    if (ret == -1) {
        std::cerr << "Error creating directory nsg_graph" << std::endl;
        return 1;
    }

    if ((dir = opendir("cluster_data")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            std::smatch matches;
            if (std::regex_match(filename, matches, pattern)) {
                int cluster_id = std::stoi(matches[1]);
                
                // 加载cluster数据
                std::string cluster_filename = "cluster_data/" + filename;
                float* cluster_data = nullptr;
                unsigned points_num, dim;
                load_data(cluster_filename, cluster_data, points_num, dim);

                std::cout << "Building NSG for cluster " << cluster_id 
                          << " with " << points_num << " points" << std::endl;

                // 构建NSG
                efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
                efanna2e::Parameters paras;
                paras.Set<unsigned>("L", L);
                paras.Set<unsigned>("R", R);
                paras.Set<unsigned>("C", C);
                paras.Set<std::string>("nn_graph_path", "nndescent/nndescent_" + std::to_string(cluster_id) + ".graph");

                std::cout << "NSG parameters for cluster " << cluster_id << ":" << std::endl
                          << "  L: " << L << std::endl
                          << "  R: " << R << std::endl
                          << "  C: " << C << std::endl;

                try {
                    auto start_time = std::chrono::high_resolution_clock::now();
                    index.Build(points_num, cluster_data, paras);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> build_time = end_time - start_time;
                    std::cout << "Successfully built NSG for cluster " << cluster_id 
                              << " in " << build_time.count() << " seconds" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error building NSG for cluster " << cluster_id << ": " << e.what() << std::endl;
                    delete[] cluster_data;
                    continue;
                }

                // 保存NSG
                std::string nsg_filename = "nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
                try {
                    index.Save(nsg_filename.c_str());
                    std::cout << "Successfully saved NSG to " << nsg_filename << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Error saving NSG for cluster " << cluster_id << ": " << e.what() << std::endl;
                    delete[] cluster_data;
                    continue;
                }

                delete[] cluster_data;
            }
        }
        closedir(dir);
    }

    return 0;
}
