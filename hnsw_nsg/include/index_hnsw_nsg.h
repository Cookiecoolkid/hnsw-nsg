#include "efanna2e/index_nsg.h"
#include "hnswlib/hnswalg.h"

namespace hnsw_nsg {

using namespace hnswlib;
using namespace efanna2e;

template<typename dist_t>
class HierarchicalNSG : public AlgorithmInterface<dist_t> {
private:
    // NSG相关数据结构
    CompactGraph nsg_graph_;
    unsigned nsg_ep_;
    size_t nsg_width_;
    Parameters nsg_params_;
    
    // HNSW原有结构扩展
    std::vector<CompactGraph> layers_; // 每层的NSG图
    std::vector<unsigned> entry_points_; // 每层入口点

public:
    HierarchicalNSW(SpaceInterface<dist_t>* s, const Parameters& params) 
        : nsg_params_(params) {
        // 初始化参数
        nsg_width_ = params.Get<unsigned>("R");
        nsg_ep_ = 0;
    }

    // 重写连接建立函数
    tableint mutuallyConnectNewElement(
        const void* data_point, tableint cur_c, 
        std::priority_queue<std::pair<dist_t, tableint>>& top_candidates,
        int level, bool isUpdate) override {
        
        // 使用NSG的同步剪枝策略
        boost::dynamic_bitset<> flags(this->max_elements_, 0);
        std::vector<Neighbor> pool;
        std::vector<Neighbor> tmp;

        // 获取NSG风格的邻居
        this->get_nsg_neighbors(data_point, level, flags, tmp, pool);

        // 执行NSG同步剪枝
        SimpleNeighbor* cut_graph = new SimpleNeighbor[this->max_elements_ * nsg_width_];
        this->sync_prune_nsg(cur_c, pool, flags, cut_graph);

        // 建立互连
        this->InterInsertNSG(cur_c, cut_graph);

        delete[] cut_graph;
        return select_entry_point(pool);
    }

    // NSG风格的邻居获取
    void get_nsg_neighbors(const void* query, int level, 
                          boost::dynamic_bitset<>& flags,
                          std::vector<Neighbor>& tmp,
                          std::vector<Neighbor>& pool) {
        // 实现NSG的候选集生成逻辑
        unsigned L = nsg_params_.Get<unsigned>("L");
        // ... NSG搜索逻辑 ...
    }

    // NSG同步剪枝
    void sync_prune_nsg(tableint q, std::vector<Neighbor>& pool,
                      boost::dynamic_bitset<>& flags,
                      SimpleNeighbor* cut_graph) {
        // 实现NSG剪枝逻辑
        unsigned range = nsg_params_.Get<unsigned>("R");
        unsigned maxc = nsg_params_.Get<unsigned>("C");
        // ... NSG剪枝逻辑 ...
    }

    // NSG互连插入
    void InterInsertNSG(tableint n, SimpleNeighbor* cut_graph) {
        // 实现NSG互连逻辑
        std::vector<std::mutex> locks(this->max_elements_);
        unsigned range = nsg_params_.Get<unsigned>("R");
        // ... NSG互连逻辑 ...
    }

    // 重写搜索逻辑
    std::priority_queue<std::pair<dist_t, labeltype>> 
    searchKnn(const void* query_data, size_t k, 
             BaseFilterFunctor* filter = nullptr) const override {
        
        std::priority_queue<std::pair<dist_t, labeltype>> results;
        
        // 分层NSG搜索
        for(int level = this->maxlevel_; level >= 0; --level) {
            // 获取当前层入口点
            unsigned ep = entry_points_[level];
            
            // 执行NSG风格搜索
            std::vector<unsigned> candidates = 
                nsg_search_layer(query_data, ep, level, k);
            
            // 合并结果
            for(auto& cand : candidates) {
                float dist = this->fstdistfunc_(
                    query_data, 
                    this->getDataByInternalId(cand),
                    this->dist_func_param_);
                results.emplace(dist, this->getExternalLabel(cand));
            }
        }
        return results;
    }

    // NSG分层搜索
    std::vector<unsigned> nsg_search_layer(const void* query, 
                                          unsigned entry_point,
                                          int level,
                                          size_t k) const {
        std::vector<unsigned> results;
        const CompactGraph& layer = layers_[level];
        
        // 实现NSG搜索逻辑
        unsigned visited_cnt = 0;
        std::priority_queue<Neighbor> candidate_set;
        boost::dynamic_bitset<> visited(this->max_elements_);
        
        candidate_set.emplace(entry_point, 
            this->fstdistfunc_(query, 
                this->getDataByInternalId(entry_point),
                this->dist_func_param_));

        while(!candidate_set.empty() && visited_cnt++ < nsg_params_.Get<unsigned>("L_search")) {
            auto curr = candidate_set.top();
            candidate_set.pop();

            if(visited[curr.id]) continue;
            visited.set(curr.id);
            results.push_back(curr.id);

            for(unsigned neighbor : layer[curr.id]) {
                float dist = this->fstdistfunc_(
                    query, 
                    this->getDataByInternalId(neighbor),
                    this->dist_func_param_);
                candidate_set.emplace(neighbor, dist);
            }
        }
        return results;
    }

    // 重写构建逻辑
    void BuildHierarchy() {
        // 初始化多层NSG结构
        for(int level = 0; level <= this->maxlevel_; ++level) {
            // 构建每层NSG图
            CompactGraph layer_graph;
            build_nsg_layer(level, layer_graph);
            layers_.push_back(layer_graph);
            
            // 确定入口点
            entry_points_[level] = find_layer_entry_point(level);
        }
    }

    void build_nsg_layer(int level, CompactGraph& graph) {
        // 实现NSG分层构建逻辑
        unsigned range = nsg_params_.Get<unsigned>("R");
        graph.resize(this->max_elements_);
        
        #pragma omp parallel for
        for(size_t i = 0; i < this->cur_element_count; ++i) {
            // 获取分层数据
            const void* data_point = this->getDataByInternalId(i);
            
            // NSG图构建流程
            std::vector<Neighbor> pool;
            boost::dynamic_bitset<> flags(this->max_elements_);
            
            // 生成候选集
            this->get_nsg_neighbors(data_point, level, flags, pool);
            
            // 剪枝优化
            sync_prune_nsg(i, pool, flags, graph.data());
        }
    }
};

// NSG参数初始化
Parameters InitializeNSGParams() {
    Parameters params;
    params.Set<unsigned>("L", 200);   // 搜索深度
    params.Set<unsigned>("R", 50);    // 邻居数
    params.Set<unsigned>("C", 500);   // 候选数
    params.Set<unsigned>("L_search", 50); // 搜索迭代次数
    return params;
}

} // namespace hnsw_nsg