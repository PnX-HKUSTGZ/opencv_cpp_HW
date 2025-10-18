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

### 更自由的操作:直接访问单个像素的各个通道

在 C++ 中，直接访问和修改 `cv::Mat` 对象的像素是核心操作之一。最安全、最常用的方法是使用 `.at<T>(y, x)` 模板函数。

-   **T**: 像素的数据类型。
    -   对于**灰度图** (`CV_8UC1`)，类型是 `uchar` (无符号8位字符)。
    -   对于**BGR彩色图** (`CV_8UC3`)，类型是 `cv::Vec3b`。`cv::Vec3b` 是一个包含3个 `uchar` 的向量，分别对应 Blue, Green, Red 通道。
-   **(y, x)**: 像素的坐标，**注意是 `(行, 列)`，即 `(row, col)`**，这与我们通常习惯的 `(x, y)` 相反。

**示例代码：**

```cpp
// 创建一个3x3的BGR彩色图像
cv::Mat image(3, 3, CV_8UC3, cv::Scalar(0, 0, 0));

// 访问位于 (行=1, 列=2) 的像素
cv::Vec3b& pixel = image.at<cv::Vec3b>(1, 2);

// 读取该像素的BGR值
uchar blue = pixel[0];
uchar green = pixel[1];
uchar red = pixel[2];

std::cout << "Pixel at (1, 2) - B: " << (int)blue << ", G: " << (int)green << ", R: " << (int)red << std::endl;

// 修改该像素的值 (例如，设置为黄色 B=0, G=255, R=255)
pixel[0] = 0;
pixel[1] = 255;
pixel[2] = 255;

std::cout << "After modification, B: " << (int)image.at<cv::Vec3b>(1, 2)[0] << std::endl;
```

**注意**: `.at<>()` 方法会进行边界检查，如果访问越界会抛出异常，因此相对安全。但在需要极高性能的循环中，直接通过指针（如 `image.ptr<uchar>(y)`）访问会更快，但操作时需要用户自己保证不会越界。


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

#### 注意: 什么是fourcc?

`fourcc` 是 "Four-Character Code" (四字符代码) 的缩写。它是一个4字节（32位）的代码，用于唯一标识视频的**编码格式 (Codec)**。

当您使用 `cv::VideoWriter` 创建视频文件时，必须告诉它使用哪种压缩算法来保存视频帧。`fourcc` 就是用来指定这个算法的。

`int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');`

这行代码的含义是：
-   调用 `cv::VideoWriter::fourcc()` 函数。
-   传入四个字符：'X', 'V', 'I', 'D'。
-   函数会将这四个字符的ASCII码拼接成一个32位的整数，这个整数就是 XVID 编码格式的标识符。

**常见的 FourCC 值:**
-   `cv::VideoWriter::fourcc('M', 'J', 'P', 'G')`: Motion-JPEG 编码，兼容性好，但文件体积较大。
-   `cv::VideoWriter::fourcc('X', 'V', 'I', 'D')`: XVID 编码，是一种常见的 MPEG-4 编码，压缩率较高。
-   `cv::VideoWriter::fourcc('H', '2', '6', '4')`: H.264 编码，目前非常流行的编码格式，压缩率很高。
-   `cv::VideoWriter::fourcc('D', 'I', 'V', 'X')`: DIVX 编码，也是一种 MPEG-4 编码。

**重要提示**: 您能否成功使用某种编码，取决于您的操作系统上是否安装了对应的解码器/编码器。如果 `VideoWriter` 创建失败，更换 `fourcc` 是常见的排错方法之一。

## 5. 图像处理

### 图像预处理

#### 颜色空间转换 (`cv::cvtColor`)

在图像处理中，我们不仅限于BGR颜色空间。根据任务的不同，转换到其他颜色空间可能会非常有用。最常见的转换是转为灰度图，但HSV、HLS等空间在颜色检测等场景下也极其重要。

-   **灰度 (Grayscale)**: 简化了图像信息，将三通道的颜色信息压缩为单通道的亮度信息。适用于许多不依赖颜色的分析，如形状和运动分析。
-   **HSV/HLS**:
    -   **H**: Hue (色相)，表示颜色本身（如红、绿、蓝）。
    -   **S**: Saturation (饱和度)，表示颜色的纯度或深浅。
    -   **V/L**: Value/Lightness (明度)，表示颜色的明亮程度。
    这个空间非常符合人类对颜色的感知。将颜色（H）与光照强度（V）分离，使得在不同光照条件下稳定地追踪特定颜色成为可能。例如，要追踪一个红色物体，在HSV空间中设定一个H通道的范围会比在BGR空间中设定一个复杂的R,G,B范围要鲁棒得多。

**C++ 代码:**
```cpp
// 转换到灰度图
cv::Mat gray_image;
cv::cvtColor(bgr_image, gray_image, cv::COLOR_BGR2GRAY);

// 转换到HSV空间
cv::Mat hsv_image;
cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);
```

#### 图像平滑与滤波 (`cv::GaussianBlur`)

图像滤波是图像处理中最基本和常用的操作之一，其主要目的是根据需求修改或增强图像，例如降噪、锐化或边缘提取。

##### 核心概念：卷积 (Convolution)

几乎所有的滤波操作都是基于“卷积”这一数学概念。在图像处理中，可以将其理解为一个**核 (Kernel)** 在图像上滑动并计算的过程。

-   **核 (Kernel)**: 一个小的矩阵（例如 3x3, 5x5），其中心点被称为“锚点”。核内的数值定义了滤波器的特性。
-   **过程**: 将核的锚点对准图像中的某个像素，核覆盖的区域内的所有像素值与核中对应位置的数值相乘，然后将所有乘积相加，得到的结果将替换掉锚点对应的那个像素的新值。这个过程对图像中的每一个像素都重复一遍。

##### 常见滤波器

**1. 高斯模糊 (Gaussian Blur)**

高斯模糊使用一个高斯核，核中心的权重最大，越远离中心的权重越小。这使得它在降噪的同时能较好地保留图像的整体轮廓，是最常用的模糊方法。

-   `cv::GaussianBlur(src, dst, ksize, sigmaX)`
    -   `ksize`: 高斯核的大小，必须是正奇数，如 `cv::Size(5, 5)`。
    -   `sigmaX`: 高斯核在X方向上的标准差。标准差越大，模糊程度越高。如果设为0，则会根据核大小自动计算。

**2. 中值滤波 (Median Filtering)**

中值滤波对于处理“椒盐噪声”（图像中随机出现的黑白像素点）特别有效。它将核覆盖区域内的所有像素值进行排序，然后用中间值（中位数）来替换中心像素的值。因为它不进行加权平均，所以能很好地去除离群像素点，同时比高斯模糊更能保留边缘的清晰度。

-   `cv::medianBlur(src, dst, ksize)`
    -   `ksize`: 核的大小，一个大于1的奇数整数。

##### 自定义核滤波

除了OpenCV内置的滤波器，你还可以定义自己的核来实现特定的滤波效果，如锐化、边缘检测等。这通过 `cv::filter2D` 函数实现。

例如，我们可以创建一个“十字形”的核来进行滤波：
```
0  1  0
1 -4  1
0  1  0
```
这个核实际上是一个拉普拉斯算子，可以用来检测边缘。

**C++ 代码:**
```cpp
// 高斯模糊
cv::Mat blurred_image;
cv::GaussianBlur(image, blurred_image, cv::Size(5, 5), 0);

// 中值滤波
cv::Mat median_blurred_image;
cv::medianBlur(image, median_blurred_image, 5);

// 自定义核 (一个简单的锐化核)
cv::Mat kernel = (cv::Mat_<float>(3,3) <<
     0, -1,  0,
    -1,  5, -1,
     0, -1,  0);
cv::Mat sharpened_image;
cv::filter2D(image, sharpened_image, -1, kernel);
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

`cv::threshold(src, dst, thresh, maxval, type)`

-   `src`: 输入图像，通常是单通道灰度图。
-   `dst`: 输出的二值图像。
-   `thresh`: 设定的阈值。
-   `maxval`: 当像素值超过阈值时（或满足特定类型条件时）赋予的值，通常是255。
-   `type`: 阈值操作的类型，这是理解二值化的关键。

##### 二值化类型 (`type`)

| 类型 | 描述 |
| :--- | :--- |
| `cv::THRESH_BINARY` | 如果 `src(x,y) > thresh`，则 `dst(x,y) = maxval`；否则 `dst(x,y) = 0`。 |
| `cv::THRESH_BINARY_INV` | `cv::THRESH_BINARY` 的反转。如果 `src(x,y) > thresh`，则 `dst(x,y) = 0`；否则 `dst(x,y) = maxval`。 |
| `cv::THRESH_TRUNC` | 截断。如果 `src(x,y) > thresh`，则 `dst(x,y) = thresh`；否则 `dst(x,y) = src(x,y)`。（像素值上限被设为阈值） |
| `cv::THRESH_TOZERO` | 如果 `src(x,y) > thresh`，则 `dst(x,y) = src(x,y)`；否则 `dst(x,y) = 0`。（低于阈值的像素归零） |
| `cv::THRESH_TOZERO_INV` | `cv::THRESH_TOZERO` 的反转。 |

##### 自动阈值：大津算法 (Otsu's Method)

在很多情况下，手动设置一个固定的阈值效果不佳，因为光照等条件会变化。大津算法是一种自动寻找最优全局阈值的方法。它通过最大化类间方差来找到一个能最好地将像素分为前景和背景两类的阈值。

要使用大津算法，只需将 `thresh` 参数设为0，并在 `type` 参数中额外加上 `cv::THRESH_OTSU` 标志。

**C++ 代码:**
```cpp
cv::Mat gray_image, binary_image;
// ... 将原图转为灰度图 gray_image ...

// 固定阈值
cv::threshold(gray_image, binary_image, 127, 255, cv::THRESH_BINARY);

// 大津算法
cv::Mat otsu_binary_image;
cv::threshold(gray_image, otsu_binary_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
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
#### 注意: 什么是Canny边缘检测?

Canny 边缘检测是一种非常流行且效果出色的边缘检测算法。它不是简单的一步操作，而是包含了一系列图像处理步骤，旨在精确地定位边缘，同时抑制噪声。

其基本原理可以分为以下四个主要步骤：

**第一步：高斯模糊 (Noise Reduction)**
-   **目的**: 消除图像中的噪声。噪声可能会在后续步骤中被错误地识别为边缘。
-   **方法**: 使用一个高斯滤波器对原始图像进行卷积操作，平滑图像，去除孤立的噪声点。

**第二步：计算图像梯度 (Gradient Calculation)**
-   **目的**: 找到图像中像素强度变化最显著的地方，这些地方很可能是边缘。
-   **方法**: 使用 Sobel 算子（一种卷积核）分别计算图像在水平（Gx）和垂直（Gy）方向上的梯度。根据这两个方向的梯度，可以计算出每个像素点的**梯度强度**（代表边缘的“强度”）和**梯度方向**（代表边缘的方向）。梯度强度越大的地方，越有可能是边缘。

**第三步：非极大值抑制 (Non-maximum Suppression)**
-   **目的**: 将第二步得到的“模糊”边缘细化成单像素宽度的“清晰”边缘。
-   **方法**: 遍历所有像素，检查每个像素在其梯度方向上的邻近像素。如果当前像素的梯度强度不是其梯度方向上（前后两个邻居）的最大值，那么就将该像素的梯度强度置为0。这样一来，只有局部最强的点才会被保留下来，从而形成细线状的边缘。

**第四步：双阈值迟滞处理 (Hysteresis Thresholding)**
-   **目的**: 确定哪些边缘是真正的边缘，哪些是因噪声或颜色变化引起的伪边缘。
-   **方法**: 这一步使用两个阈值：`minVal` (低阈值) 和 `maxVal` (高阈值)。
    1.  梯度强度**高于 `maxVal`** 的像素点被确定为“强边缘”，是最终边缘的一部分。
    2.  梯度强度**低于 `minVal`** 的像素点被直接舍弃，不视为边缘。
    3.  梯度强度**介于 `minVal` 和 `maxVal` 之间**的像素点被标记为“弱边缘”。只有当一个“弱边缘”像素连接到一个“强边缘”像素时（直接或通过其他弱边缘间接连接），它才会被保留为最终边缘的一部分。

通过这种方式，Canny 算法能够有效地连接由弱边缘组成的断裂边缘，同时去除那些孤立的、由噪声引起的弱边缘点，最终得到清晰、连续的边缘图像。

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

找到轮廓后，我们可以计算它们的各种特征或用几何形状去拟合它们。

-   **面积**: `cv::contourArea(contour)`
-   **周长**: `cv::arcLength(contour, true)` (第二个参数 `true` 表示轮廓是闭合的)

**几何形状拟合:**

-   **多边形逼近 (Polygon Approximation)**: 使用 `cv::approxPolyDP` 将轮廓近似成一个顶点较少的多边形。这对于简化轮廓形状很有用。
-   **最小外接矩形 (Minimum Area Rectangle)**: 使用 `cv::minAreaRect` 找到包围轮廓的最小面积的矩形（这个矩形可以是旋转的）。返回一个 `cv::RotatedRect` 对象。
-   **最小外接圆 (Minimum Enclosing Circle)**: 使用 `cv::minEnclosingCircle` 找到包围轮廓的最小圆。
-   **最小外接三角形 (Minimum Enclosing Triangle)**: 使用 `cv::minEnclosingTriangle` 找到包围轮廓的最小三角形。
-   **直线拟合**: 使用 `cv::fitLine` 将一个点集拟合成一条直线。

**C++ 代码:**
```cpp
for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    if (area > 100) {
        double perimeter = cv::arcLength(contour, true);
        cv::Rect bounding_box = cv::boundingRect(contour);
        
        // 最小外接旋转矩形
        cv::RotatedRect rotatedRect = cv::minAreaRect(contour);

        // 多边形逼近
        std::vector<cv::Point> approx_poly;
        cv::approxPolyDP(contour, approx_poly, perimeter * 0.02, true);
    }
}
```

### 形态学操作

形态学操作是基于形状的图像处理技术，通常作用于二值图像。它们使用一个被称为**结构元素 (Structuring Element)** 的小核来探测和修改图像的形状。

**结构元素**: 类似于卷积核，你可以指定它的形状（矩形、十字形、椭圆形）和大小。`cv::getStructuringElement()` 函数用于创建它。

-   **腐蚀 (Erosion)**: "腐蚀"掉物体边界的像素，使物体变小。可以用来消除小的噪声点，或者分离两个连接在一起的物体。
-   **膨胀 (Dilation)**: "膨胀"物体边界的像素，使物体变大。可以用来填充物体内部的孔洞，或者连接两个断开的物体。

基于腐蚀和膨胀，可以组合出更高级的操作：

-   **开运算 (Opening)**: 先腐蚀后膨胀。主要作用是消除小的噪点（小白点），平滑物体轮廓，并且在不明显改变物体面积的情况下断开细小的连接。
-   **闭运算 (Closing)**: 先膨胀后腐蚀。主要作用是填充物体内部的小孔洞（小黑点），连接邻近的物体。
-   **形态学梯度 (Morphological Gradient)**: 膨胀图减去腐蚀图。可以得到物体的轮廓。
-   **顶帽 (Top Hat)**: 原图减去开运算图。可以获得图像中的噪声或者比邻近区域更亮的细节。
-   **黑帽 (Black Hat)**: 闭运算图减去原图。可以获得图像内部的小孔，或者比邻近区域更暗的细节。

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

动手实践 `src/main.cpp` 中的代码，并尝试修改它来实现不同的功能。

## 8. 作业/应用: 灯条检测

基于 RoboMaster 比赛的背景，灯条检测是视觉识别任务的第一步，其目的是在复杂的背景中准确地找到装甲板两侧的高亮灯条。以下是完成该任务的简明思路流程：

1.  **图像预处理 (Preprocessing)**
    *   **目的**: 消除图像噪声，为后续处理做准备。
    *   **方法**: 使用 `cv::GaussianBlur` 对图像进行轻微的模糊处理。

2.  **提取颜色与亮度特征 (Feature Extraction)**
    *   **目的**: 将灯条从复杂的背景中凸显出来。这是关键一步。
    *   **可选方法**:
        *   **方法一**: **通道相减**。根据目标颜色（红/蓝），将对应颜色通道减去另一个通道 (`B-R` 或 `R-B`)。这能非常有效地分离出特定颜色的高亮区域。
        *   **方法二**: **转灰度图**。如果不考虑颜色，只利用灯条的高亮度特性，可以直接将图像转为灰度图。

3.  **二值化 (Binarization)**
    *   **目的**: 将上一步得到的灰度图像转换为只有黑白两种颜色的二值图像，将灯条区域变为白色，背景变为黑色。
    *   **方法**: 使用 `cv::threshold`，设置一个合适的阈值。

4.  **寻找轮廓 (Contour Finding)**
    *   **目的**: 从二值图像中找出所有白色物体的边界。
    *   **方法**: 使用 `cv::findContours` 函数。

5.  **筛选轮廓 (Filtering Contours)**
    *   **目的**: 剔除噪声和非目标物体，找到“候选灯条”。
    *   **步骤**:
        1.  **几何特征筛选**: 遍历所有轮廓，计算其**最小外接矩形** (`cv::minAreaRect`)。根据矩形的**长宽比**、**面积**和**倾斜角度**进行筛选。
        2.  **颜色判断**: 对通过几何筛选的轮廓，通过 `cv::pointPolygonTest` 确定其内部像素，判断其真实颜色（红/蓝）。

6.  **配对灯条 (Matching Light Bars)**
    *   **目的**: 将单个的灯条配对成一个完整的装甲板。
    *   **方法**: 遍历所有“候选灯条”，两两组合。根据配对灯条之间的**角度差**、**高度差**、**距离**等几何约束，找到最有可能组成一个装甲板的灯条对。

7.  **输出结果 (Output)**
    *   成功配对的灯条即为最终识别到的装甲板。

[text](https://sugary-honeycup-41c.notion.site/eca6035db1034f0a98fdbfbcce92cd86?v=57f3e63d063a457a9c109446eca8a0db&p=67daf03af7fd4bfd987d91c85be3a19e&pm=s)
这是去年的相关详细教程.


