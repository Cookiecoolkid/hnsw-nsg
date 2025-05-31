#include "aux_util.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace CNNS {

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

} // namespace CNNS
