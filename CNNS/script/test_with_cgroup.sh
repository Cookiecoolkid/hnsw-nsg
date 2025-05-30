#!/bin/bash

CGROUP_DIR="/sys/fs/cgroup/mymemgroup"
MEM_LIMIT="max"
SWAP_LIMIT="max"
#TOTAL_LIMIT=$((MEM_LIMIT + SWAP_LIMIT))
MEM_HIGH=$((1024 * 1024 * 1024))
MEM_LOW=0

if [ ! -d "$CGROUP_DIR" ]; then
    sudo mkdir "$CGROUP_DIR"
fi


echo "$MEM_LIMIT" | sudo tee "$CGROUP_DIR/memory.max"
echo "$SWAP_LIMIT" | sudo tee "$CGROUP_DIR/memory.swap.max"
echo "$MEM_HIGH" | sudo tee "$CGROUP_DIR/memory.high"  # 不提前做 reclaim
echo "$MEM_LOW" | sudo tee "$CGROUP_DIR/memory.low"

# 将当前 shell 加入 cgroup
# echo $$ | sudo tee "$CGROUP_DIR/cgroup.procs"

# 启动程序
./cluster_hnsw_nsg_search_pipeline ~/datasets/sift/sift_query.fvecs ~/datasets/sift/sift_groundtruth.ivecs 20 100 100 &
PID=$!
echo $PID | sudo tee "$CGROUP_DIR/cgroup.procs"

# 初始化 I/O 次数
if [ -f "$CGROUP_DIR/io.stat" ]; then
    INITIAL_RIOS=$(grep -E 'rios' "$CGROUP_DIR/io.stat" | awk '{print $4}')
    INITIAL_WIOS=$(grep -E 'wios' "$CGROUP_DIR/io.stat" | awk '{print $5}')
fi

# 实时监控
while kill -0 $PID 2>/dev/null; do
    echo "Memory current: $(cat "$CGROUP_DIR/memory.current") bytes"
    echo "Swap current: $(cat "$CGROUP_DIR/memory.swap.current") bytes"
    
    if [ -f "$CGROUP_DIR/io.stat" ]; then
        CURRENT_RIOS=$(grep -E 'rios' "$CGROUP_DIR/io.stat" | awk '{print $4}')
        CURRENT_WIOS=$(grep -E 'wios' "$CGROUP_DIR/io.stat" | awk '{print $5}')
        
        # 计算增量
        DELTA_RIOS=$((CURRENT_RIOS - INITIAL_RIOS))
        DELTA_WIOS=$((CURRENT_WIOS - INITIAL_WIOS))
        
        echo "Read I/O operations (delta): $DELTA_RIOS"
        echo "Write I/O operations (delta): $DELTA_WIOS"
    fi
    
    echo "------"
    sleep 1
done

echo "Program has finished or was terminated."
