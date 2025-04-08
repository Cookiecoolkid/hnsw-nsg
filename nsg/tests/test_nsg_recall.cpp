#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <algorithm>

// 读取二进制文件
std::vector<std::vector<unsigned>> loadBinaryFile(const char* filename) {
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

// 计算召回率
float calculateRecall(const std::vector<std::vector<unsigned>>& result, const std::vector<std::vector<unsigned>>& groundtruth) {
    size_t correct = 0;
    size_t total = 0;

    for (size_t i = 0; i < result.size(); ++i) {
        std::unordered_set<unsigned> gt_set(groundtruth[i].begin(), groundtruth[i].end());
        total += gt_set.size();

        for (unsigned id : result[i]) {
            if (gt_set.find(id) != gt_set.end()) {
                correct++;
            }
        }
    }

    return 1.0f * correct / total;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <result_file> <groundtruth_file>" << std::endl;
        return 1;
    }

    const char* result_file = argv[1];
    const char* groundtruth_file = argv[2];

    try {
        std::vector<std::vector<unsigned>> result = loadBinaryFile(result_file);
        std::vector<std::vector<unsigned>> groundtruth = loadBinaryFile(groundtruth_file);

        if (result.size() != groundtruth.size()) {
            std::cerr << "Error: Result and groundtruth files have different number of queries." << std::endl;
            return 1;
        }

        float recall = calculateRecall(result, groundtruth);
        std::cout << "Recall: " << recall << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}