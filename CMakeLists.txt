# ============================================================================
# CMakeLists.txt - 统一 TensorRT 推理测试
# ============================================================================
# 
# 构建步骤:
#   mkdir build && cd build
#   cmake ..
#   make -j$(nproc)
#   ./unified_trt_test
#
# 依赖:
#   - CUDA Toolkit (>= 11.0)
#   - TensorRT (>= 8.0, 推荐 10.x)
#
# 环境变量 (可选):
#   - CUDA_HOME: CUDA 安装路径
#   - TENSORRT_ROOT: TensorRT 安装路径
#
# ============================================================================

cmake_minimum_required(VERSION 3.18)
project(unified_trt_test LANGUAGES CXX CUDA)

# ============================================================================
# C++ 标准
# ============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# 查找 CUDA
# ============================================================================
find_package(CUDA REQUIRED)

# CUDA 包含目录
include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

# ============================================================================
# 查找 TensorRT
# ============================================================================
# 尝试从环境变量或常见路径查找 TensorRT
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")

# 常见的 TensorRT 安装路径
set(TENSORRT_SEARCH_PATHS
    ${TENSORRT_ROOT}
    $ENV{TENSORRT_ROOT}
    /usr/local/tensorrt
    /usr/local/TensorRT
    /opt/tensorrt
    /usr/include/x86_64-linux-gnu
    /usr/lib/x86_64-linux-gnu
)

# 查找 TensorRT 头文件
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    PATHS ${TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES include
)

# 查找 TensorRT 库
find_library(TENSORRT_LIBRARY_INFER
    NAMES nvinfer
    PATHS ${TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TENSORRT_LIBRARY_PLUGIN
    NAMES nvinfer_plugin
    PATHS ${TENSORRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR)
    message(FATAL_ERROR "TensorRT include directory not found. Please set TENSORRT_ROOT.")
endif()

if(NOT TENSORRT_LIBRARY_INFER)
    message(FATAL_ERROR "TensorRT library (nvinfer) not found. Please set TENSORRT_ROOT.")
endif()

message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library (nvinfer): ${TENSORRT_LIBRARY_INFER}")
message(STATUS "TensorRT Library (nvinfer_plugin): ${TENSORRT_LIBRARY_PLUGIN}")

include_directories(${TENSORRT_INCLUDE_DIR})

# ============================================================================
# 项目源文件
# ============================================================================
# 包含当前目录 (unified_trt.h 所在位置)
include_directories(${CMAKE_SOURCE_DIR})

# ============================================================================
# 可执行文件: unified_trt_test
# ============================================================================
add_executable(unified_trt_test
    unified_main.cpp
)

# 链接库
target_link_libraries(unified_trt_test
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_PLUGIN}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# ============================================================================
# 编译选项
# ============================================================================
# 优化选项
target_compile_options(unified_trt_test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
)

# ============================================================================
# 安装规则 (可选)
# ============================================================================
install(TARGETS unified_trt_test
    RUNTIME DESTINATION bin
)

install(FILES unified_trt.h
    DESTINATION include
)

# ============================================================================
# 打印配置信息
# ============================================================================
message(STATUS "")
message(STATUS "=============== Configuration Summary ===============")
message(STATUS "Project:            ${PROJECT_NAME}")
message(STATUS "C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Include:       ${CUDA_INCLUDE_DIRS}")
message(STATUS "TensorRT Include:   ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT nvinfer:   ${TENSORRT_LIBRARY_INFER}")
message(STATUS "=====================================================")
message(STATUS "")
message(STATUS "Build commands:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make -j$(nproc)")
message(STATUS "  ./unified_trt_test")
message(STATUS "")
