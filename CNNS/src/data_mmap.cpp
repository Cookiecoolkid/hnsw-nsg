#include "data_mmap.h"
#include <iostream>
#include <sys/stat.h>

namespace CNNS {

bool load_cluster_data_mmap(
    int cluster_id,
    unsigned global_dim,
    ClusterMMap& cluster_info,
    const std::string& prefix) {
    
    std::string filename = prefix + "/cluster_data/cluster_" + std::to_string(cluster_id) + ".fvecs";
    
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open cluster file " << filename << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    size_t file_size = sb.st_size;
    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    const char* ptr = reinterpret_cast<const char*>(mapped);
    int dim;
    std::memcpy(&dim, ptr, sizeof(int));
    if ((unsigned)dim != global_dim) {
        std::cerr << "Dimension mismatch: expected " << global_dim << ", got " << dim << std::endl;
        munmap(mapped, file_size);
        close(fd);
        return false;
    }

    size_t vec_size_bytes = sizeof(int) + dim * sizeof(float);
    size_t points_num = file_size / vec_size_bytes;

    cluster_info.data_ptr = reinterpret_cast<float*>(const_cast<char*>(ptr) + sizeof(int));
    cluster_info.points_num = points_num;
    cluster_info.dim = dim;
    cluster_info.mmap_base = mapped;
    cluster_info.mmap_length = file_size;
    cluster_info.fd = fd;
    return true;
}

bool load_id_mapping_mmap(
    int cluster_id,
    unsigned points_num,
    MappingMMap& mapping_info,
    const std::string& prefix) {

    std::string filename = prefix + "/mapping/mapping_" + std::to_string(cluster_id);
    int fd = open(filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Cannot open mapping file " << filename << std::endl;
        return false;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "fstat failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    size_t file_size = sb.st_size;
    if (file_size != points_num * sizeof(faiss::idx_t)) {
        std::cerr << "Mapping file size mismatch: expected " << points_num << ", got " << file_size / sizeof(faiss::idx_t) << std::endl;
        close(fd);
        return false;
    }

    void* mapped = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        std::cerr << "mmap failed on " << filename << std::endl;
        close(fd);
        return false;
    }

    mapping_info.data = reinterpret_cast<faiss::idx_t*>(mapped);
    mapping_info.length = points_num;
    mapping_info.mmap_base = mapped;
    mapping_info.mmap_length = file_size;
    mapping_info.fd = fd;
    return true;
}

bool load_nsg_index_mmap(
    int cluster_id,
    unsigned global_dim,
    unsigned points_num,
    efanna2e::IndexNSG*& nsg_index,
    const std::string& prefix) {
    
    std::string filename = prefix + "/nsg_graph/nsg_" + std::to_string(cluster_id) + ".nsg";
    struct stat buffer;
    if (stat(filename.c_str(), &buffer) != 0) {
        std::cerr << "NSG file does not exist: " << filename << std::endl;
        return false;
    }

    nsg_index = new efanna2e::IndexNSG(global_dim, points_num, efanna2e::L2, nullptr);
    try {
        nsg_index->Load_mmap(filename.c_str());
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load NSG index: " << e.what() << std::endl;
        delete nsg_index;
        nsg_index = nullptr;
        return false;
    }
}

void load_cluster_specific_data_and_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, ClusterMMap>& cluster_data_map,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix) {

    try {
        // 使用显式构造函数加载数据
        cluster_data_map[cluster_id] = ClusterMMap(cluster_id, global_dim, prefix);
        id_mapping_map[cluster_id] = MappingMMap(cluster_id, cluster_data_map[cluster_id].points_num, prefix);
        
        // 加载NSG索引
        efanna2e::IndexNSG* nsg_index = nullptr;
        if (!load_nsg_index_mmap(cluster_id, global_dim, cluster_data_map[cluster_id].points_num, nsg_index, prefix)) {
            throw std::runtime_error("Failed to load NSG index");
        }
        cluster_nsg_indices[cluster_id] = nsg_index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading cluster " << cluster_id << ": " << e.what() << std::endl;
        // 清理已加载的资源
        cluster_data_map.erase(cluster_id);
        id_mapping_map.erase(cluster_id);
        if (cluster_nsg_indices[cluster_id]) {
            delete cluster_nsg_indices[cluster_id];
            cluster_nsg_indices.erase(cluster_id);
        }
    }
}

} // namespace CNNS