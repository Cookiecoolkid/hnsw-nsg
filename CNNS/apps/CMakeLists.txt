# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/apps)

# 获取所有cpp文件
file(GLOB APP_SOURCES "*.cpp")

# 为每个源文件创建可执行文件
foreach(SOURCE_FILE ${APP_SOURCES})
    # 获取文件名（不含扩展名）
    get_filename_component(EXECUTABLE_NAME ${SOURCE_FILE} NAME_WE)
    
    # 创建可执行文件
    add_executable(${EXECUTABLE_NAME} ${SOURCE_FILE})
    
    # 链接必要的库
    target_link_libraries(${EXECUTABLE_NAME}
        ${FAISS_LIB}
        ${BLAS_LIB}
        ${OpenMP_CXX_LIBRARIES}
    )
    
    # 设置包含目录
    target_include_directories(${EXECUTABLE_NAME} PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/apps/include
    )
endforeach()

# 安装规则
install(TARGETS ${EXECUTABLE_NAME}
        RUNTIME DESTINATION bin) 