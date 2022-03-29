#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

void gradientSobel() {
  cv::Mat img = cv::imread("../images/img1.png");
  std::string img_window = "original image";
  cv::namedWindow(img_window, 1);
  cv::imshow(img_window, img);

  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);
  cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

  // apply gaussian blur
  float gauss_data[25] = {1,  4, 7, 4,  1,  4,  16, 26, 16, 4, 7, 26, 41,
                          26, 7, 4, 16, 26, 16, 4,  1,  4,  7, 4, 1};
  for (int i = 0; i < 25; i++) {
    gauss_data[i] /= 273;
  }
  cv::Mat kernel_gauss = cv::Mat(5, 5, CV_32F, gauss_data);

  std::vector<cv::Mat> kernels = {kernel_gauss, kernel_x, kernel_y};
  for (auto kernel : kernels) {
    cv::Mat img_sobel;
    cv::filter2D(img_gray, img_sobel, -1, kernel, cv::Point(-1, -1), 0,
                 cv::BORDER_DEFAULT);
    std::string img_sobel_window = "filtered image";
    cv::namedWindow(img_sobel_window, 1);
    cv::imshow(img_sobel_window, img_sobel);
  }

  cv::waitKey(0);
}

int main() { gradientSobel(); }
