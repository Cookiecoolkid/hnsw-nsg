#ifndef AUX_UTIL_H
#define AUX_UTIL_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <memory>
#include <thread>
#include <mutex>
#include <omp.h>

namespace CNNS {

// 加载fvecs文件
std::vector<float> load_fvecs(const std::string& filename, unsigned& num, unsigned& dim);

// 加载ground truth文件
std::vector<std::vector<unsigned>> loadGT(const char* filename);

// 加载质心文件
std::vector<float> load_centroids(const std::string& filename, int& n_clusters, int& m, unsigned& dim);

// 计算L2距离
float compute_l2_distance(const float* a, const float* b, unsigned dim);

// 计算余弦相似度
float compute_cosine_similarity(const float* a, const float* b, unsigned dim);

// 计算内积
float compute_inner_product(const float* a, const float* b, unsigned dim);

// 计算Cosine距离
float compute_cosine_distance(const float* a, const float* b, unsigned dim);

} // namespace CNNS

#endif // AUX_UTIL_H
