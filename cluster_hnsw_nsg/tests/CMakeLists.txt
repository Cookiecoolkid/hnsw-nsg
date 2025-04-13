# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 测试程序1
add_executable(test_build test_build.cpp)
target_link_libraries(test_build 
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)

# test_faissIVF需要链接更多FAISS组件
add_executable(test_faissIVF test_faissIVF.cpp)
target_link_libraries(test_faissIVF
    ${FAISS_LIB}
    ${OpenBLAS_LIB}
    pthread
    m
    ${OpenMP_CXX_FLAGS}
)