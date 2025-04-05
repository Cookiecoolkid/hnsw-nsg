#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <vector>
#include <unordered_set>
#include <string>
#include <cmath>

#include "../../hnswlib/hnswlib.h"

using namespace std;
using namespace hnswlib;

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

static void get_gt(
    unsigned int *massQA,
    float *massQ,
    float *mass,
    size_t vecsize,
    size_t qsize,
    L2Space &l2space,
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers,
    size_t k) {
    
    answers.resize(qsize);
    for (size_t i = 0; i < qsize; ++i) {
        for (size_t j = 0; j < k; ++j) {
            answers[i].emplace(0.0f, massQA[i * 100 + j]);
        }
    }
}


static float test_approx(
    float *massQ,  // 修改为float*
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg, // 修改为float
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers, // 修改为float
    size_t k) {
    
    size_t correct = 0;
    size_t total = 0;
    for (int i = 0; i < qsize; i++) {
        auto result = appr_alg.searchKnn(massQ + vecdim * i, k);
        auto gt = answers[i];
        unordered_set<labeltype> g;
        total += gt.size();

        while (!gt.empty()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (!result.empty()) {
            // cout << result.top().first << "\t" << result.top().second << "\n";
            if (g.find(result.top().second) != g.end()) {
                correct++;
                // cout << "correct\n";
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void test_vs_recall(
    float *massQ,  // 修改为float*
    size_t vecsize,
    size_t qsize,
    HierarchicalNSW<float> &appr_alg, // 修改为float
    size_t vecdim,
    vector<std::priority_queue<std::pair<float, labeltype>>> &answers, // 修改为float
    size_t k) {
    
    vector<size_t> efs;
    for (int i = k; i < 30; i++) efs.push_back(i);
    for (int i = 30; i < 100; i += 10) efs.push_back(i);
    for (int i = 100; i < 500; i += 40) efs.push_back(i);

    auto total_time = 0.0f;
    cout << "EF\tRecall\tTime(us)" << endl;  // 新增表头
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        StopW stopw;
        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;
        cout << ef << "\t" << recall << "\t" << time_us_per_query << " us\n";
        total_time += stopw.getElapsedTimeMicro();  // 累加总时间
        if (recall > 1.0) break;
    }
    // 输出总时间(seconds)
    cout << "Total time for all queries: " << total_time / 1e6 << " seconds" << endl;
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
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

// [保留原有test_approx和test_vs_recall实现]

void sift_test1M() {
    const int efConstruction = 40;
    const int M = 16;
    const size_t vecdim = 128;

    // 使用绝对路径！！！
    const char *path_q = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_query.fvecs";
    const char *path_data = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_base.fvecs";
    const char *path_gt = "/home/cookiecoolkid/Research/Graph-Computation/hnsw-nsg/dataset/sift/sift_groundtruth.ivecs";
    char path_index[128];
    snprintf(path_index, sizeof(path_index), "sift1M_ef%d_M%d.bin", efConstruction, M);

    // 加载基础数据集
    float* data_load = nullptr;
    unsigned points_num, dim;
    load_data(path_data, data_load, points_num, dim);
    cout << "Loaded " << points_num << " points, dim: " << dim << endl;
    cout << "Sample data[0][0:3]: " 
         << data_load[0] << ", " << data_load[1] << ", " << data_load[2] << endl;

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

    // 初始化HNSW索引
    L2Space space(vecdim);
    HierarchicalNSW<float>* appr_alg = nullptr;

    chrono::duration<double> build_time{0};  // 新增：构建时间统计变量

    if (exists_test(path_index)) {
        cout << "Loading index from: " << path_index << endl;
        appr_alg = new HierarchicalNSW<float>(&space, path_index);
    } else {
        cout << "Building index..." << endl;
        auto build_start = chrono::high_resolution_clock::now();  // 记录构建开始时间

        appr_alg = new HierarchicalNSW<float>(&space, points_num, M, efConstruction);

        StopW stopw;  // 用于测量处理时间
        size_t report_every = 100000;  // 每处理 100,000 个点输出一次信息

        // 添加数据点
        #pragma omp parallel for
        for (size_t i = 0; i < points_num; ++i) {
            appr_alg->addPoint(data_load + i * vecdim, i);
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

        auto build_end = chrono::high_resolution_clock::now();  // 记录构建结束时间
        build_time = build_end - build_start;  // 计算构建耗时

        appr_alg->saveIndex(path_index);
        cout << "Index build time: " << build_time.count() << " seconds" << endl;  // 输出构建时间
    }

    // 验证召回率
    vector<std::priority_queue<std::pair<float, labeltype>>> answers;
    get_gt(gt, query_load, nullptr, points_num, query_num, space, vecdim, answers, 100);
    test_vs_recall(query_load, points_num, query_num, *appr_alg, vecdim, answers, 100);

    // 清理资源
    delete[] data_load;
    delete[] query_load;
    delete[] gt;
    delete appr_alg;
}
