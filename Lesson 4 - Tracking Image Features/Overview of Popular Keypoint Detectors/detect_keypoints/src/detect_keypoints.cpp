#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void detKeypoints1() {
  // load image from file and convert to grayscale
  cv::Mat imgGray;
  cv::Mat img = cv::imread("../images/img1.png");
  cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

  // Shi-Tomasi detector
  int block_size = 6;
  // in percent
  double max_overlap = 0.0;
  double min_distance = (1.0 - max_overlap) * block_size;
  int max_keypoints = img.rows * img.cols / std::max(1.0, min_distance);
  double quality_level = 0.01;
  double k = 0.04;
  bool use_harris = false;

  // TODO(a-ngo): write a function that takes different detectors

  // shi tomasi detector
  std::vector<cv::KeyPoint> kptsShiTomasi;
  std::vector<cv::Point2f> corners;
  double t = static_cast<double>(cv::getTickCount());
  cv::goodFeaturesToTrack(imgGray, corners, max_keypoints, quality_level,
                          min_distance, cv::Mat(), block_size, use_harris, k);

  t = static_cast<double>(cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "Shi-Tomasi with n= " << corners.size() << " keypoints in "
            << 1000 * t / 1.0 << " ms" << std::endl;

  for (auto it = corners.begin(); it != corners.end(); ++it) {
    // add corners to result vector
    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = block_size;
    kptsShiTomasi.push_back(newKeyPoint);
  }

  // visualize results
  cv::Mat visImage = img.clone();
  cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  std::string windowName = "Shi-Tomasi Results";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, visImage);

  // FAST detector
  std::vector<cv::KeyPoint> fast_keypoints;
  std::vector<cv::Point2f> fast_corners;

  t = static_cast<double>(cv::getTickCount());
  double threshold{75};
  bool non_max_suppression{false};
  cv::FAST(imgGray, fast_keypoints, threshold, non_max_suppression);

  t = static_cast<double>(cv::getTickCount() - t) / cv::getTickFrequency();
  std::cout << "FAST with n= " << fast_keypoints.size() << " keypoints in "
            << 1000 * t / 1.0 << " ms" << std::endl;

  // visualize results
  cv::Mat visImage_fast = img.clone();
  cv::drawKeypoints(img, fast_keypoints, visImage_fast, cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  windowName = "FAST Results";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, visImage_fast);

  // use the OpenCV library to add the FAST detector
  // in addition to the already implemented Shi-Tomasi
  // detector and compare both algorithms with regard to
  // (a) number of keypoints, (b) distribution of
  // keypoints over the image and (c) processing speed.
  cv::waitKey(0);
}

int main() { detKeypoints1(); }
