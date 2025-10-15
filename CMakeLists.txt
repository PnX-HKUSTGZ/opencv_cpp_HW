
cmake_minimum_required(VERSION 3.10)
project(opencv_cpp_demo)

# 查找OpenCV包
find_package(OpenCV REQUIRED)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(opencv_cpp_demo src/main.cpp)

# 链接OpenCV库
target_link_libraries(opencv_cpp_demo ${OpenCV_LIBS})
