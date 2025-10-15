A guidence for Noob to learn opencv in C++ Language
# OpenCV C++ 教案

本教案旨在将 Python 版本的 OpenCV 教程翻译为 C++ 版本，并提供一个完整的 C++ OpenCV 项目结构，帮助你快速上手。

## 1. C++ 环境配置与项目构建

与 Python 不同，C++ 需要一个编译环境。我们使用 `CMake` 来管理项目，因为它跨平台且被广泛支持。

### 安装 OpenCV for C++

首先，你需要在你的系统上安装 OpenCV 库。

- **Ubuntu/Debian/WSL**:
  ```bash
  sudo apt-get update
  sudo apt-get install libopencv-dev
  ```
- **macOS (使用 Homebrew)**:
  ```bash
  brew install opencv
  ```
- **Windows**:
  从 [OpenCV 官网](https://opencv.org/releases/) 下载预编译好的库，并配置系统环境变量。或者使用 `vcpkg` 进行安装。

### 使用 CMake 构建项目

我们提供了一个 `CMakeLists.txt` 文件来构建本项目。

1.  **创建 build 文件夹**:
    ```bash
    mkdir build
    cd build
    ```
2.  **运行 CMake**:
    ```bash
    cmake ..
    ```
3.  **编译项目**:
    ```bash
    make
    ```
4.  **运行可执行文件**:
    ```bash
    ./opencv_cpp_demo
    ```

---

## 2. 计算机眼中的图片

这部分理论与 Python 版本完全相同。图像的本质是像素、坐标和颜色。

### 图像的本质——像素、坐标与颜色

- **像素 (Pixel)**: 构成数字图像的最小单位。
- **坐标系 (Coordinate System)**: OpenCV C++ 中同样使用左上角为原点 `(0, 0)`，X 轴向右，Y 轴向下的坐标系。
- **颜色与通道 (Color & Channels)**:
    - **灰度图**: 单通道图像。
    - **彩色图**: OpenCV C++ 默认使用 **BGR** 顺序。
    - **带透明度图**: BGRA 四通道图像。
    - **RGB-D**: 包含颜色信息和深度信息。

### 图像就是 `cv::Mat`

在 C++ 中，图像被表示为 `cv::Mat` 对象。`cv::Mat` 是一个强大的多维数组类，是 OpenCV C++ 的核心。

- **灰度图**: 一个 `cv::Mat` 对象，其形状为 `(height, width)`，类型为 `CV_8UC1` (8位无符号单通道)。
- **彩色图**: 一个 `cv::Mat` 对象，其形状为 `(height, width)`，类型为 `CV_8UC3` (8位无符号三通道)。

---

## 3. 读取、写入和显示

### 读取图像 (`cv::imread`)

```cpp
#include <opencv2/opencv.hpp>
#include <string>

// ...
std::string filepath = "images/RGBD-sample.jpg";
cv::Mat image = cv::imread(filepath, cv::IMREAD_COLOR);
```
- `filepath`: 图像文件的路径。
- `flags`:
    - `cv::IMREAD_COLOR` (默认): BGR 彩色图。
    - `cv::IMREAD_GRAYSCALE`: 灰度图。
    - `cv::IMREAD_UNCHANGED`: 读取所有通道，包括 Alpha。

### 保存图像 (`cv::imwrite`)

```cpp
cv::imwrite("output.png", image);
```
- 第一个参数是输出路径，扩展名决定了压缩格式。
- 第二个参数是要保存的 `cv::Mat` 对象。

### 显示图像 (`cv::imshow`, `cv::waitKey`, `cv::destroyAllWindows`)

```cpp
cv::imshow("Window Title", image);
cv::waitKey(0); // 无限等待按键
cv::destroyAllWindows();
```
- `cv::imshow`: 在窗口中显示图像。
- `cv::waitKey(delay)`: 暂停程序。`0` 表示无限等待，`>0` 表示等待指定毫秒数。
- `cv::destroyAllWindows`: 关闭所有 OpenCV 窗口。

### 示例：读写和显示

```cpp
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // --- 1. 读取 ---
    std::string img_path = "./images/RGBD-sample.jpg";
    cv::Mat image = cv::imread(img_path);

    // --- 2. 检查 ---
    if (image.empty()) {
        std::cerr << "错误: 无法在路径 '" << img_path << "' 找到图片。" << std::endl;
        return -1;
    }

    // --- 3. 显示 ---
    std::string window_title = "RobotMaster Demo";
    cv::imshow(window_title, image);
    std::cout << "图片已在窗口 '" << window_title << "' 中显示。按任意键关闭。" << std::endl;
    
    cv::waitKey(0);
    cv::destroyAllWindows();
    std::cout << "窗口已关闭。" << std::endl;

    // --- 4. 处理与写入 ---
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    
    std::string output_path = "./images/RGBD-sample-gray.png";
    cv::imwrite(output_path, gray_image);
    std::cout << "处理后的图像已保存到 '" << output_path << "'。" << std::endl;

    return 0;
}
```

---

## 4. 视频流处理

### 读取视频 (`cv::VideoCapture`)

```cpp
cv::VideoCapture cap(0); // 从默认摄像头读取
// 或者 cv::VideoCapture cap("my_video.mp4");

if (!cap.isOpened()) {
    // 错误处理
}

cv::Mat frame;
while (cap.read(frame)) {
    // 处理每一帧
}
cap.release();
```

### 写入视频 (`cv::VideoWriter`)

```cpp
int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
cv::Size frame_size(w, h);
int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
double fps = 20.0;

cv::VideoWriter writer("output.avi", fourcc, fps, frame_size);

// 在循环中
writer.write(frame);

writer.release();
```

---

## 5. 图像处理

### 图像预处理

#### 颜色空间转换 (`cv::cvtColor`)

```cpp
cv::Mat gray_image;
cv::cvtColor(bgr_image, gray_image, cv::COLOR_BGR2GRAY);
```

#### 图像平滑与滤波 (`cv::GaussianBlur`)

```cpp
cv::Mat blurred_image;
cv::GaussianBlur(image, blurred_image, cv::Size(5, 5), 0);
```
- `cv::Size(5, 5)` 是核大小，必须是正奇数。

### 通道分离与合并

#### `cv::split`

```cpp
std::vector<cv::Mat> channels;
cv::split(image_bgr, channels);
// channels[0] 是蓝色通道, channels[1] 是绿色, channels[2] 是红色
cv::Mat b = channels[0];
cv::Mat g = channels[1];
cv::Mat r = channels[2];
```

#### `cv::merge`

```cpp
cv::Mat merged_image;
std::vector<cv::Mat> new_channels = {b, g, r};
cv::merge(new_channels, merged_image);
```

---

## 6. 图像分割与特征提取

### 图像阈值处理

#### `cv::threshold` (全局阈值)

```cpp
cv::Mat binary_image;
cv::threshold(grayscale_image, binary_image, 127, 255, cv::THRESH_BINARY);
```
- `127` 是阈值。
- `255` 是最大值。
- `cv::THRESH_BINARY` 是二值化类型。

#### `cv::adaptiveThreshold` (自适应阈值)

```cpp
cv::Mat adaptive_thresh_image;
cv::adaptiveThreshold(grayscale_image, adaptive_thresh_image, 255, 
                      cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                      cv::THRESH_BINARY, 11, 2);
```
- `cv::ADAPTIVE_THRESH_GAUSSIAN_C`: 高斯加权平均法。
- `11`: 邻域大小 (block_size)。
- `2`: 常数 C。

### 边缘检测 (`cv::Canny`)

```cpp
cv::Mat edges;
cv::Canny(blurred_gray_image, edges, 50, 150);
```
- `50` 是低阈值 `threshold1`。
- `150` 是高阈值 `threshold2`。

### 轮廓分析

#### `cv::findContours`

```cpp
std::vector<std::vector<cv::Point>> contours;
std::vector<cv::Vec4i> hierarchy;
cv::findContours(binary_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
```
- `contours`: 存储所有发现的轮廓。
- `hierarchy`: 存储轮廓的层级关系。
- `cv::RETR_EXTERNAL`: 只检测最外层轮廓。
- `cv::CHAIN_APPROX_SIMPLE`: 压缩轮廓点。

#### `cv::drawContours`

```cpp
cv::drawContours(output_image, contours, -1, cv::Scalar(0, 255, 0), 2);
```
- `-1`: 绘制所有轮廓。
- `cv::Scalar(0, 255, 0)`: 颜色 (绿色)。
- `2`: 线条粗细。

#### 轮廓属性分析

```cpp
for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true); // true 表示闭合
    cv::Rect bounding_box = cv::boundingRect(contour);
    
    if (area > 100) {
        // 进行分析...
    }
}
```

### 形态学操作

#### `cv::erode` (腐蚀) 和 `cv::dilate` (膨胀)

```cpp
cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

cv::Mat eroded_image;
cv::erode(binary_image, eroded_image, kernel);

cv::Mat dilated_image;
cv::dilate(binary_image, dilated_image, kernel);
```

#### `cv::morphologyEx` (开/闭运算)

```cpp
// 开运算: 去除外部噪点
cv::Mat opening_image;
cv::morphologyEx(binary_image, opening_image, cv::MORPH_OPEN, kernel);

// 闭运算: 填充内部孔洞
cv::Mat closing_image;
cv::morphologyEx(binary_image, closing_image, cv::MORPH_CLOSE, kernel);
```

---

## 7. 结语

本教案涵盖了 OpenCV C++ 的基础操作。与 Python 版本相比，C++ 版本在语法上更严谨，需要手动管理内存（虽然 `cv::Mat` 做了很多智能管理），并且需要编译。但其性能优势在处理大规模图像和实时视频流时非常明显。

鼓励你动手实践 `src/main.cpp` 中的代码，并尝试修改它来实现不同的功能。
