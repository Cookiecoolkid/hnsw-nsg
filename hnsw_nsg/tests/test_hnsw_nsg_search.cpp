#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <string>
#include <cmath>

#include "index_hnsw_nsg.h"  // 包含混合索引头文件

using namespace std;
using namespace hnsw_nsg;

// [保留原有StopW、内存统计函数、get_gt、test_approx、test_vs_recall、load_data等工具函数]
class StopW {
    std::chrono::steady_clock::time_point time_begin;
 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}



// 修改后的 test_approx 函数（适配 HNSW_NSG）
static float test_approx(float *massQ, size_t vecsize, size_t qsize, HNSW_NSG<float> &hybrid_alg, // 修改为混合索引类型
    size_t vecdim, vector<vector<unsigned>> &answers, size_t k) {
    
    size_t correct = 0;
    size_t total = 0;
    for (size_t i = 0; i < qsize; i++) {
        // 调用混合索引的搜索接口
        auto result = hybrid_alg.searchKnn(massQ + vecdim * i, k, nullptr);
        auto gt = answers[i];
        unordered_set<unsigned> g;
        total += gt.size();

        for (auto id : gt) {
            g.insert(id);
        }

        for (auto id : result) {
            if (g.find(id) != g.end()) {
                correct++;
            }
        }
    }
    return 1.0f * correct / total;
}


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

// 修改后的 test_vs_recall 函数（适配 HNSW_NSG）
static void test_vs_recall(
    float *massQ,
    size_t vecsize,
    size_t qsize,
    HNSW_NSG<float> &hybrid_alg, // 修改为混合索引类型
    size_t vecdim,
    vector<vector<unsigned>> &answers,
    size_t k) {
    
    vector<size_t> efs;
    // for (int i = k; i < 30; i++) efs.push_back(i);
    // for (int i = 30; i < 100; i += 10) efs.push_back(i);
    // for (int i = 100; i < 500; i += 40) efs.push_back(i);

    efs.push_back(k);

    cout << "EF\tRecall\tTime(us)" << endl;
    for (size_t ef : efs) {
        // 设置 HNSW 的 ef 参数（需在 HNSW_NSG 类中添加对应方法）
        hybrid_alg.setEf(ef);
        StopW stopw = StopW();
        float recall = test_approx(massQ, vecsize, qsize, hybrid_alg, vecdim, answers, k);
        float total_time = stopw.getElapsedTimeMicro();
        float time_us_per_query = total_time / qsize;
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n" << "total time: " << total_time / 1e6 << " seconds" << endl;
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

bool exists_tests(const std::string &name, const std::string &hnsw_path, const std::string &nsg_path) {
    std::string hnsw_name = name + hnsw_path;
    std::string nsg_name = name + nsg_path;
    return exists_test(hnsw_name) && exists_test(nsg_name);
}

void load_data(const char* filename, float*& data, unsigned& num, unsigned& dim) {
    ifstream in(filename, ios::binary);
    if (!in.is_open()) {
        cerr << "Open file error: " << filename << endl;
        exit(1);
    }
    
    // 读取维度
    in.read((char*)&dim, 4);
    
    // 计算数据点数量
    in.seekg(0, ios::end);
    size_t fsize = in.tellg();
    num = fsize / ((dim + 1) * 4);
    
    // 分配内存
    data = new float[num * dim];
    
    // 读取数据
    in.seekg(0, ios::beg);
    for (size_t i = 0; i < num; ++i) {
        in.seekg(4, ios::cur); // 跳过维度头
        in.read((char*)(data + i * dim), dim * sizeof(float));
    }
    in.close();
}


void sift_test1M(Parameters &params) {
    const int efConstruction = 40;
    const int M = 16;
    const unsigned nsg_width = 20;  // NSG参数
    const size_t vecdim = 128;

    // 使用绝对路径！！！
    const char *path_q = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_query.fvecs";
    const char *path_data = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_base.fvecs";
    const char *path_gt = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_groundtruth.ivecs";
    char path_index[128];
    snprintf(path_index, sizeof(path_index), "sift1M_hybrid_ef%d_M%d_nsgw%d", efConstruction, M, nsg_width);

    // 加载数据集（与原始测试相同）
    float* data_load = nullptr;
    unsigned points_num, dim;
    load_data(path_data, data_load, points_num, dim);
    cout << "Loaded " << points_num << " points, dim: " << dim << endl;

    // 加载查询集
    float* query_load = nullptr;
    unsigned query_num, query_dim;
    load_data(path_q, query_load, query_num, query_dim);
    cout << "Loaded " << query_num << " queries" << endl;

    // 加载Ground Truth
    ifstream gt_input(path_gt, ios::binary);
    unsigned int* gt = new unsigned int[query_num * 100];
    for (size_t i = 0; i < query_num; ++i) {
        int t;
        gt_input.read((char*)&t, sizeof(int));
        if (t != 100) {
            cerr << "Invalid GT format at query " << i << "and t = " << t << endl;
            exit(1);
        }
        gt_input.read((char*)(gt + i*100), 100 * sizeof(unsigned int));
    }
    gt_input.close();

    // 初始化混合索引
    L2Space space(vecdim);
    HNSW_NSG<float>* hybrid_alg = nullptr;  // 使用混合索引

    chrono::duration<double> build_time{0};
    std::string path_hnsw = "_hnsw.bin";
    std::string path_nsg = "_nsg.bin";

    if (exists_tests(path_index, path_hnsw, path_nsg)) {
        cout << "Loading index from path: " << path_index << endl;
        hybrid_alg = new HNSW_NSG<float>(&space, vecdim, points_num, data_load, query_load, params, path_index + path_hnsw, path_index + path_nsg);
    } else {
        cout << "Building hybrid index..." << endl;
        auto build_start = chrono::high_resolution_clock::now();

        // 初始化混合索引
        hybrid_alg = new HNSW_NSG<float>(&space, dim, points_num, data_load, query_load, params, M, efConstruction, nsg_width);

        StopW stopw;
        size_t report_every = 100000;

        // 添加数据点（自动处理映射）
        #pragma omp parallel for
        for (size_t i = 0; i < points_num; ++i) {
            hybrid_alg->addPoint(data_load + i * vecdim, i, false);
            
            if (i % report_every == 0) {
                auto elapsed_time = stopw.getElapsedTimeMicro();  // 获取当前处理时间
                double kips = report_every / (elapsed_time / 1e6);  // 计算 kips
                double mem_usage = getCurrentRSS() / 1000000.0;  // 获取当前内存使用量（单位：MB）
                cout << "Processed " << i << " points (" 
                    << i * 100.0 / points_num << "%), "
                    << "KIPS: " << kips << ", "
                    << "Mem: " << mem_usage << " Mb" << endl;
                stopw.reset();  // 重置计时器
            }
        }

        hybrid_alg->Build_NSG(data_load, params);  // 构建NSG图

        auto build_end = chrono::high_resolution_clock::now();
        build_time = build_end - build_start;
        hybrid_alg->saveIndex(path_index);  // 保存索引
        cout << "Hybrid index build time: " << build_time.count() << " seconds" << endl;
    }

    // 验证召回率（使用混合索引搜索）
    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    std::vector<std::vector<unsigned>> groundtruth = loadBinaryFile(path_gt);
    test_vs_recall(query_load, points_num, query_num, *hybrid_alg, vecdim, groundtruth, 100);

    // 内存统计
    cout << "Peak RSS: " << getPeakRSS() / 1000000 << " MB" << endl;

    // 清理资源
    delete[] data_load;
    delete[] query_load;
    delete[] gt;
    delete hybrid_alg;
}

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cout << argv[0] << " nn_graph_path L R C save_graph_file search_L search_K result_path"
                << std::endl;
        exit(-1);
    }

    std::string nn_graph_path(argv[1]);
    unsigned L = (unsigned)atoi(argv[2]);
    unsigned R = (unsigned)atoi(argv[3]);
    unsigned C = (unsigned)atoi(argv[4]);
    unsigned search_L = (unsigned)atoi(argv[6]);
    unsigned search_K = (unsigned)atoi(argv[7]);

    Parameters params;
    params.Set<unsigned>("L", L);
    params.Set<unsigned>("R", R);
    params.Set<unsigned>("C", C);
    params.Set<std::string>("nn_graph_path", nn_graph_path);
    params.Set<unsigned>("L_search", search_L);
    params.Set<unsigned>("P_search", search_L);
    params.Set<unsigned>("K_search", search_K);

    std::string result_path(argv[8]);
    params.Set<std::string>("result_path", result_path);
    sift_test1M(params);
    return 0;
}