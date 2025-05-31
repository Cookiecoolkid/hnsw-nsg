#pragma once

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/index_io.h>
#include <efanna2e/index_graph.h>
#include <efanna2e/index_random.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "nsg/neighbor.h"
#include "nsg/index_nsg.h"
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace CNNS {

class ClusterMMap {
public:
    ClusterMMap() = delete;
    
    ClusterMMap(int cluster_id, unsigned global_dim, const std::string& prefix) {
        if (!load_cluster_data_mmap(cluster_id, global_dim, *this, prefix)) {
            throw std::runtime_error("Failed to load cluster data");
        }
    }

    ~ClusterMMap() {
        if (mmap_base && mmap_base != MAP_FAILED) munmap(mmap_base, mmap_length);
        if (fd != -1) close(fd);
    }

    float* data_ptr = nullptr;
    size_t points_num = 0;
    size_t dim = 0;
    void* mmap_base = nullptr;
    size_t mmap_length = 0;
    int fd = -1;
};

class MappingMMap {
public:
    MappingMMap() = delete;
    
    MappingMMap(int cluster_id, unsigned points_num, const std::string& prefix) {
        if (!load_id_mapping_mmap(cluster_id, points_num, *this, prefix)) {
            throw std::runtime_error("Failed to load mapping data");
        }
    }

    ~MappingMMap() {
        if (mmap_base && mmap_base != MAP_FAILED) munmap(mmap_base, mmap_length);
        if (fd != -1) close(fd);
    }

    faiss::idx_t* data = nullptr;
    size_t length = 0;
    void* mmap_base = nullptr;
    size_t mmap_length = 0;
    int fd = -1;
};

bool load_cluster_data_mmap(
    int cluster_id,
    unsigned global_dim,
    ClusterMMap& cluster_info,
    const std::string& prefix);

bool load_id_mapping_mmap(
    int cluster_id,
    unsigned points_num,
    MappingMMap& mapping_info,
    const std::string& prefix);

bool load_nsg_index_mmap(
    int cluster_id,
    unsigned global_dim,
    unsigned points_num,
    efanna2e::IndexNSG*& nsg_index,
    const std::string& prefix);

void load_cluster_specific_data_and_nsg_mmap(
    int cluster_id,
    unsigned global_dim,
    std::map<int, ClusterMMap>& cluster_data_map,
    std::map<int, MappingMMap>& id_mapping_map,
    std::map<int, efanna2e::IndexNSG*>& cluster_nsg_indices,
    const std::string& prefix);

} // namespace CNNS