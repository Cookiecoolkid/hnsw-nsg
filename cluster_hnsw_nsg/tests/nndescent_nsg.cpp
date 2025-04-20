#include <index_nsg.h>
#include <util.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <dirent.h>
#include <regex>

// 加载fvecs文件
void load_data(const std::string& filename, float*& data, unsigned& num, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[(size_t)num * (size_t)dim];

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
        return 1;
    }

    // 创建nsg_graph目录
    system("mkdir -p nsg_graph");

    // NSG参数
    unsigned L = (unsigned)atoi(argv[1]); // 40
    unsigned R = (unsigned)atoi(argv[2]); // 50
    unsigned C = (unsigned)atoi(argv[3]); // 默认为 500

    // 遍历nndescent目录下的所有文件
    DIR* dir;
    struct dirent* ent;
    std::regex pattern("nndescent_(\\d+)\\.graph");
    std::smatch matches;

    if ((dir = opendir("nndescent")) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string filename = ent->d_name;
            if (std::regex_match(filename, matches, pattern)) {
                int centroid_id = std::stoi(matches[1]);

                // 加载对应的cluster数据
                std::string cluster_filename = "nndescent/" + filename;
                float* cluster_data = NULL;
                unsigned points_num, dim;
                load_data(cluster_filename, cluster_data, points_num, dim);

                // 构建NSG图
                efanna2e::IndexNSG index(dim, points_num, efanna2e::L2, nullptr);
                
                efanna2e::Parameters paras;
                paras.Set<unsigned>("L", L);
                paras.Set<unsigned>("R", R);
                paras.Set<unsigned>("C", C);
                paras.Set<std::string>("nn_graph_path", "nndescent/" + filename);

                std::cout << "Building NSG for cluster " << centroid_id << "..." << std::endl;
                auto s = std::chrono::high_resolution_clock::now();
                index.Build(points_num, cluster_data, paras);
                auto e = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> diff = e - s;
                std::cout << "Indexing time: " << diff.count() << " seconds" << std::endl;

                // 保存NSG图
                std::string nsg_filename = "nsg_graph/nsg_" + std::to_string(centroid_id) + ".nsg";
                index.Save(nsg_filename.c_str());

                delete[] cluster_data;
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open nndescent directory" << std::endl;
        return 1;
    }

    return 0;
}
