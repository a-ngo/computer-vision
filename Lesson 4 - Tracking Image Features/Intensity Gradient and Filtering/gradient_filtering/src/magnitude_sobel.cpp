#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void magnitudeSobel() {
  // load image from file
  cv::Mat img = cv::imread("../images/img1gray.png");
  std::string img_window = "original image";
  cv::namedWindow(img_window, 1);
  cv::imshow(img_window, img);

  // convert image to grayscale
  cv::Mat img_gray;
  cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

  // apply smoothing using the GaussianBlur() function from the OpenCV
  cv::Mat img_gaussian = img_gray.clone();
  int filterSize = 5;
  int stdDev = 2.0;
  cv::GaussianBlur(img_gray, img_gaussian, cv::Size(filterSize, filterSize),
                   stdDev);

  // create filter kernels using the cv::Mat datatype both for x and y
  float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

  cv::Mat kernel_x = cv::Mat(3, 3, CV_32F, sobel_x);
  cv::Mat kernel_y = cv::Mat(3, 3, CV_32F, sobel_y);

  // apply filter using the OpenCv function filter2D()
  cv::Mat img_sobel_x, img_sobel_y;
  cv::filter2D(img_gaussian, img_sobel_x, -1, kernel_x, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);
  cv::filter2D(img_gaussian, img_sobel_y, -1, kernel_y, cv::Point(-1, -1), 0,
               cv::BORDER_DEFAULT);

  // compute magnitude image based on the equation presented in the lesson
  cv::Mat magnitude = img_gray.clone();
  std::cout << "rows: " << magnitude.rows << std::endl;
  std::cout << "cols: " << magnitude.cols << std::endl;

  for (int r = 0; r < magnitude.rows; r++) {
    for (int c = 0; c < magnitude.cols; c++) {
      magnitude.at<unsigned char>(r, c) =
          sqrt(pow(img_sobel_x.at<unsigned char>(r, c), 2) +
               pow(img_sobel_y.at<unsigned char>(r, c), 2));
    }
  }

  // show result
  std::string windowName = "Sobel filter magnitude image";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, magnitude);
  cv::waitKey(0);
}

int main() { magnitudeSobel(); }
