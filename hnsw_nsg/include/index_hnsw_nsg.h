#include "nsg/index_nsg.h"
#include "hnsw/hnswalg.h"

namespace hnsw_nsg {

template<typename dist_t>
class HNSW_NSG : public AlgorithmInterface<dist_t> {
public:
    //////////////////////////////
    // 内嵌的原始结构实例 //
    //////////////////////////////
    HierarchicalNSW<dist_t> hnsw_;
    IndexNSG nsg_;

    float* data_;
    float* query_data_;
    size_t dim_;
    Parameters parameters_;
    unsigned point_num_;

    std::vector<std::vector<unsigned>> res;

    //////////////////////////////
    // 跨结构映射关系 //
    //////////////////////////////

    // 似乎不需要，直接分别构造加载，搜索时判读即可

    // std::vector<tableint> hnsw_to_nsg_;  // HNSW节点到NSG节点的映射
    // std::vector<tableint> nsg_to_hnsw_;  // NSG节点到HNSW节点的映射

public:
    //////////////////////////////
    // 构造函数（独立初始化） //
    //////////////////////////////
    HNSW_NSG(SpaceInterface<dist_t>* space, unsigned int dim, size_t max_elements, float* data, float* query_data,
             const Parameters& parameters, unsigned hnsw_M = 16, size_t hnsw_ef_construction = 200, 
             unsigned nsg_width = 20) 
             : hnsw_(space, max_elements, hnsw_M, hnsw_ef_construction, 100, false), 
               nsg_(dim,max_elements, L2, nullptr) {
        data_ = data;
        query_data_ = query_data;
        dim_ = dim;
        parameters_ = parameters;
        point_num_ = max_elements;
    }

    //////////////////////////////
    // 析构函数（独立释放） //
    //////////////////////////////
    ~HNSW_NSG() {}

    void setEf(size_t ef) {
        hnsw_.ef_ = ef;
    }

    void Build_NSG(const float* data, Parameters& parameters) {
        nsg_.Build(point_num_, data, parameters);
    }

    //////////////////////////////
    // 插入逻辑（双向维护） //
    //////////////////////////////
    void addPoint(const void* data_point, labeltype label, bool replace_deleted) override {
        // Step 1: 插入HNSW（所有层）
        hnsw_.addPoint(data_point, label, replace_deleted);
    }
    
    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchHybridLayer(tableint enterpoint, const void* query_data, size_t K, 
                      BaseFilterFunctor* isIdAllowed) const {
        unsigned id = enterpoint;

        // 执行NSG搜索
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> nsg_result;
        std::vector<unsigned> indices(K);
        const float *query_float = static_cast<const float*>(query_data);
        // nsg_.SearchFromEnterpoint(query_float, data_ ,K, parameters_, indices.data(), id);
        nsg_.MySearch(query_float, data_, K, parameters_, indices.data());

        res.push_back(indices);

        for (size_t i = 0; i < indices.size(); i++) {
            unsigned id = indices[i];
            dist_t dist = hnsw_.fstdistfunc_(query_data, hnsw_.getDataByInternalId(id), hnsw_.dist_func_param_);
            nsg_result.emplace(dist, id);
        }

        return nsg_result;
    }
    //////////////////////////////
    // 搜索逻辑（分层处理） //
    //////////////////////////////
    std::priority_queue<std::pair<dist_t, labeltype>> 
    searchKnn(const void* query, size_t k, BaseFilterFunctor* filter) const override {
        // Phase 1: HNSW上层搜索（L1及以上层）
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (hnsw_.cur_element_count == 0) return result;

        tableint currObj = hnsw_.enterpoint_node_;
        dist_t curdist = hnsw_.fstdistfunc_(query, hnsw_.getDataByInternalId(hnsw_.enterpoint_node_), hnsw_.dist_func_param_);

        for (int level = hnsw_.maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) hnsw_.get_linklist(currObj, level);
                int size = hnsw_.getListCount(data);
                hnsw_.metric_hops++;
                hnsw_.metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    if (cand < 0 || cand > hnsw_.max_elements_)
                        throw std::runtime_error("cand error");
                    dist_t d = hnsw_.fstdistfunc_(query, hnsw_.getDataByInternalId(cand), hnsw_.dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;

        top_candidates = searchHybridLayer(currObj, query, k, filter);

        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, hnsw_.getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    void saveIndex(const std::string& location) override {
        // 保存HNSW索引
        hnsw_.saveIndex(location + "_hnsw.bin");

        // 保存NSG索引
        nsg_.Save((location + "_nsg.bin").c_str());
    }
};
} // namespace hnsw_nsg