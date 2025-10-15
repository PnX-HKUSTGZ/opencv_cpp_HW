
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

// 函数声明
void demo_read_write_show();
void demo_video_capture();
void demo_preprocessing();
void demo_channel_operations();
void demo_thresholding();
void demo_adaptive_thresholding();
void demo_canny_edge();
void demo_contour_analysis();
void demo_morphology();

int main() {
    std::cout << "选择一个要运行的OpenCV C++ Demo:" << std::endl;
    std::cout << "1. 读写和显示图片" << std::endl;
    std::cout << "2. 捕获和保存视频" << std::endl;
    std::cout << "3. 图像预处理 (颜色转换, 模糊)" << std::endl;
    std::cout << "4. 通道分离与合并" << std::endl;
    std::cout << "5. 全局阈值处理" << std::endl;
    std::cout << "6. 自适应阈值处理" << std::endl;
    std::cout << "7. Canny 边缘检测" << std::endl;
    std::cout << "8. 轮廓分析" << std::endl;
    std::cout << "9. 形态学操作" << std::endl;
    std::cout << "输入数字 (1-9): ";

    int choice;
    std::cin >> choice;

    switch (choice) {
        case 1: demo_read_write_show(); break;
        case 2: demo_video_capture(); break;
        case 3: demo_preprocessing(); break;
        case 4: demo_channel_operations(); break;
        case 5: demo_thresholding(); break;
        case 6: demo_adaptive_thresholding(); break;
        case 7: demo_canny_edge(); break;
        case 8: demo_contour_analysis(); break;
        case 9: demo_morphology(); break;
        default:
            std::cerr << "无效的选择!" << std::endl;
            break;
    }

    return 0;
}

void demo_read_write_show() {
    std::string img_path = "../images/xiaogong.jpg";
    cv::Mat image = cv::imread(img_path);

    if (image.empty()) {
        std::cerr << "错误: 无法加载图片 " << img_path << std::endl;
        return;
    }

    cv::imshow("Original Image", image);
    cv::waitKey(0);

    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    cv::imwrite("../images/xiaogong_gray.png", gray_image);
    std::cout << "灰度图已保存。" << std::endl;

    cv::destroyAllWindows();
}

void demo_video_capture() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "错误: 无法打开摄像头。" << std::endl;
        return;
    }

    int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size size(w, h);

    cv::VideoWriter writer("output.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 20.0, size);

    std::cout << "正在录制... 按 'q' 键停止。" << std::endl;

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        writer.write(frame);
        cv::imshow("Recording...", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    std::cout << "录制结束。" << std::endl;
}

void demo_preprocessing() {
    cv::Mat image_bgr = cv::imread("../images/xiaogong.jpg");
    if (image_bgr.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    cv::Mat image_gray;
    cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred_bgr;
    cv::GaussianBlur(image_bgr, blurred_bgr, cv::Size(5, 5), 0);

    cv::Mat blurred_gray;
    cv::GaussianBlur(image_gray, blurred_gray, cv::Size(9, 9), 0);

    cv::imshow("1. Original BGR", image_bgr);
    cv::imshow("2. Blurred BGR (5x5)", blurred_bgr);
    cv::imshow("3. Grayscale", image_gray);
    cv::imshow("4. Blurred Grayscale (9x9)", blurred_gray);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_channel_operations() {
    cv::Mat image_bgr = cv::imread("../images/xiaogong.jpg");
    if (image_bgr.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    std::vector<cv::Mat> channels;
    cv::split(image_bgr, channels);

    cv::imshow("Original Image", image_bgr);
    cv::imshow("Blue Channel", channels[0]);
    cv::imshow("Green Channel", channels[1]);
    cv::imshow("Red Channel", channels[2]);
    cv::waitKey(0);

    cv::Mat merged_bgr;
    cv::merge(channels, merged_bgr);
    cv::imshow("Merged BGR", merged_bgr);

    std::vector<cv::Mat> rgb_channels = {channels[2], channels[1], channels[0]};
    cv::Mat merged_rgb;
    cv::merge(rgb_channels, merged_rgb);
    cv::imshow("Merged as RGB", merged_rgb);

    cv::Mat zeros = cv::Mat::zeros(channels[0].size(), channels[0].type());
    std::vector<cv::Mat> no_red_channels = {channels[0], channels[1], zeros};
    cv::Mat merged_no_red;
    cv::merge(no_red_channels, merged_no_red);
    cv::imshow("Image without Red", merged_no_red);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_thresholding() {
    cv::Mat image_bgr = cv::imread("../images/ceiling.jpg");
    if (image_bgr.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    cv::Mat image_gray, image_blur;
    cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image_gray, image_blur, cv::Size(5, 5), 0);

    cv::Mat binary_img, binary_inv_img;
    cv::threshold(image_blur, binary_img, 127, 255, cv::THRESH_BINARY);
    cv::threshold(image_blur, binary_inv_img, 127, 255, cv::THRESH_BINARY_INV);

    cv::imshow("Original", image_bgr);
    cv::imshow("Binary", binary_img);
    cv::imshow("Inverted Binary", binary_inv_img);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_adaptive_thresholding() {
    cv::Mat image_bgr = cv::imread("../images/adaptiveThreshold_sample.jpg");
    if (image_bgr.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    cv::Mat image_gray, image_blur;
    cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image_gray, image_blur, cv::Size(5, 5), 0);

    cv::Mat global_thresh, adaptive_mean_thresh, adaptive_gaussian_thresh;
    cv::threshold(image_blur, global_thresh, 30, 255, cv::THRESH_BINARY);
    cv::adaptiveThreshold(image_blur, adaptive_mean_thresh, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 11, 2);
    cv::adaptiveThreshold(image_blur, adaptive_gaussian_thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

    cv::imshow("Original", image_bgr);
    cv::imshow("Global Thresh", global_thresh);
    cv::imshow("Adaptive Mean", adaptive_mean_thresh);
    cv::imshow("Adaptive Gaussian", adaptive_gaussian_thresh);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_canny_edge() {
    cv::Mat image_bgr = cv::imread("../images/xiaogong.jpg");
    if (image_bgr.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    cv::Mat image_gray, image_blur;
    cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(image_gray, image_blur, cv::Size(3, 3), 0);

    cv::Mat edges, tighter_edges;
    cv::Canny(image_blur, edges, 50, 150);
    cv::Canny(image_blur, tighter_edges, 100, 250);

    cv::imshow("Original", image_bgr);
    cv::imshow("Canny (50, 150)", edges);
    cv::imshow("Tighter Canny (100, 250)", tighter_edges);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_contour_analysis() {
    cv::Mat image = cv::imread("../images/counter_sample.png");
    if (image.empty()) {
        std::cerr << "图片加载失败" << std::endl;
        return;
    }

    cv::Mat output_image = image.clone();
    cv::Mat gray, blur, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::threshold(blur, binary, 50, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = cv::contourArea(contours[i]);
        if (area < 100) continue;

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.04, true);
        size_t num_vertices = approx.size();

        std::string shape_name = "Unknown";
        cv::Rect bounding_box = cv::boundingRect(approx);

        if (num_vertices == 3) {
            shape_name = "Triangle";
            cv::rectangle(output_image, bounding_box, cv::Scalar(0, 255, 0), 2);
        } else if (num_vertices == 4) {
            double aspect_ratio = (double)bounding_box.width / bounding_box.height;
            shape_name = (aspect_ratio >= 0.95 && aspect_ratio <= 1.05) ? "Square" : "Rectangle";
            cv::rectangle(output_image, bounding_box, cv::Scalar(255, 0, 0), 2);
        } else if (num_vertices > 4) {
            shape_name = "Circle";
            cv::Point2f center;
            float radius;
            cv::minEnclosingCircle(contours[i], center, radius);
            cv::circle(output_image, center, (int)radius, cv::Scalar(0, 0, 255), 2);
        }

        cv::putText(output_image, shape_name, cv::Point(bounding_box.x, bounding_box.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }

    cv::imshow("Original", image);
    cv::imshow("Binary", binary);
    cv::imshow("Contour Analysis", output_image);

    cv::waitKey(0);
    cv::destroyAllWindows();
}

void demo_morphology() {
    cv::Mat image = cv::Mat::zeros(500, 500, CV_8U);
    cv::rectangle(image, cv::Point(100, 100), cv::Point(400, 400), 255, -1);

    for (int i = 0; i < 20; ++i) {
        int x = rand() % 400 + 50;
        int y = rand() % 400 + 50;
        if (x > 100 && x < 400 && y > 100 && y < 400) {
            cv::circle(image, cv::Point(x, y), 2, 0, -1);
        }
    }
    for (int i = 0; i < 50; ++i) {
        int x = rand() % 500;
        int y = rand() % 500;
        if (!(x > 100 && x < 400 && y > 100 && y < 400)) {
            cv::circle(image, cv::Point(x, y), 2, 255, -1);
        }
    }
    cv::Mat original_noisy = image.clone();

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    
    cv::Mat eroded, dilated, opening, closing, perfect;
    cv::erode(original_noisy, eroded, kernel);
    cv::dilate(original_noisy, dilated, kernel);
    cv::morphologyEx(original_noisy, opening, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(original_noisy, closing, cv::MORPH_CLOSE, kernel);

    cv::Mat temp;
    cv::morphologyEx(original_noisy, temp, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(temp, perfect, cv::MORPH_CLOSE, kernel);

    cv::imshow("1. Original Noisy", original_noisy);
    cv::imshow("2. Erosion", eroded);
    cv::imshow("3. Dilation", dilated);
    cv::imshow("4. Opening", opening);
    cv::imshow("5. Closing", closing);
    cv::imshow("6. Perfect", perfect);

    cv::waitKey(0);
    cv::destroyAllWindows();
}
