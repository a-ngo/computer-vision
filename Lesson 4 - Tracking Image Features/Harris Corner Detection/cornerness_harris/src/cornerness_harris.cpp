#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void cornernessHarris() {
  // load image from file
  cv::Mat img;
  img = cv::imread("../images/img1.png");
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);

  cv::namedWindow("original image", 1);
  cv::imshow("original image", img);

  // Detector parameters
  int block_size = 2;
  int aperture_size = 3;
  int min_response = 100;
  double k = 0.04;

  // Detect Harris corners and normalize output
  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img.size(), CV_32FC1);
  cv::cornerHarris(img, dst, block_size, aperture_size, k, cv::BORDER_DEFAULT);

  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

  // visualize results
  std::string windowName_norm_scaled =
      "Harris Corner Detector Response Matrix - norm scaled";
  cv::namedWindow(windowName_norm_scaled, 1);
  cv::imshow(windowName_norm_scaled, dst_norm_scaled);

  // Your task is to locate local maxima in the Harris response
  // matrix and perform a non-maximum suppression (NMS) in a local neighborhood
  // around each maximum. The resulting coordinates shall be stored in a list of
  // keypoints of the type `vector<cv::KeyPoint>`.

  std::vector<cv::KeyPoint> keypoints;

  // window size should be odd in order to center it on a pixel and have
  // symmetry in all directions
  int sliding_window_size = 7;
  int sw_dist = std::floor(sliding_window_size / 2);

  for (size_t row = 0; row < dst_norm.rows; ++row) {
    for (size_t col = 0; col < dst_norm.cols; ++col) {
      int response = static_cast<int>(dst_norm.at<float>(row, col));
      if (response > min_response) {
        // VERSION A
        // get max cornerness within local window
        for (size_t local_row = row - sw_dist; local_row <= row + sw_dist;
             local_row++) {
          for (size_t local_col = col - sw_dist; local_col <= col + sw_dist;
               local_col++) {
            if (response < dst_norm.at<unsigned int>(local_row, local_col)) {
              response = dst_norm.at<unsigned int>(local_row, local_col);
            }
          }
        }

        if (dst_norm.at<unsigned int>(row, col) != response) {
          dst_norm.at<unsigned int>(row, col) = 0;
        } else {
          cv::KeyPoint new_key_point;
          new_key_point.pt = cv::Point2f(col, row);
          new_key_point.size = 2 * aperture_size;
          new_key_point.response = response;
          keypoints.push_back(new_key_point);
        }

        // VERSION B
        // cv::KeyPoint current_key_point;
        // // TODO(a-ngo): correct? -> x,y? -> understand it
        // current_key_point.pt = cv::Point2f(col, row);
        // current_key_point.size = 2 * aperture_size;
        // current_key_point.response = response;

        // bool overlap_flag{false};
        // for (auto &keypoint : keypoints) {
        //   double overlap = cv::KeyPoint::overlap(current_key_point,
        //   keypoint);

        //   if (overlap > 0.0) {
        //     overlap_flag = true;
        //     if (keypoint.response < response) {
        //       keypoint = current_key_point;
        //       break;
        //     }
        //   }
        // }

        // if (!overlap_flag) {
        //   keypoints.push_back(current_key_point);
        // }
      }
    }
  }

  std::cout << "Number of detected corners: " << keypoints.size() << std::endl;

  std::string windowName = "Harris Corner Detector Response Matrix-final";

  cv::Mat keypoints_image = dst_norm_scaled.clone();
  cv::drawKeypoints(dst_norm_scaled, keypoints, keypoints_image,
                    cv::Scalar::all(-1),
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, keypoints_image);

  cv::waitKey(0);
}

int main() { cornernessHarris(); }
