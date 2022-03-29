#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

void showLidarTopview() {
  std::vector<LidarPoint> lidar_points;
  readLidarPts("../dat/C51_LidarPts_0000.dat", lidar_points);

  // width and height of sensor field in m
  cv::Size world_size(10.0, 20.0);

  // corresponding top view image in pixel
  cv::Size image_size(1000, 2000);

  // create topview image
  cv::Mat top_view_img(image_size, CV_8UC3, cv::Scalar(0, 0, 0));

  // plot Lidar points into image
  for (auto it = lidar_points.begin(); it != lidar_points.end(); ++it) {
    // TASK: 2. Remove all Lidar points on the road surface while
    // preserving measurements on the obstacles in the scene.
    // lidar mounting position z=+1.73m
    if ((*it).z < -1.55)
      continue;

    // x facing forward and y left from sensor in m (world position)
    float xw = (*it).x;
    float yw = (*it).y;

    int y = (-xw * image_size.height / world_size.height) + image_size.height;
    int x =
        (-yw * image_size.width / world_size.width) + image_size.width / 2.0;

    // TASK: 1. Change the color of the Lidar points such that
    // X=0.0m corresponds to red while X=20.0m is shown as green.
    int g = 0.0 + xw / 20.0 * 255.0;
    int r = 255.0 - xw / 20.0 * 255.0;

    // color coding via reflectivity
    // int g = 0.0 + (*it).r * 255.0;
    // int r = 255.0 - (*it).r * 255.0;

    cv::circle(top_view_img, cv::Point(x, y), 5, cv::Scalar(0, g, r), -1);
  }

  // plot distance markers
  float line_spacing = 2.0;
  int markers_number = floor(world_size.height / line_spacing);
  for (size_t i = 0; i < markers_number; ++i) {
    int y = (-(i * line_spacing) * image_size.height / world_size.height) +
            image_size.height;
    cv::line(top_view_img, cv::Point(0, y), cv::Point(image_size.width, y),
             cv::Scalar(255, 255, 255));
  }

  // display image
  std::string window_name = "Top-View Perspective of LiDAR data";
  cv::namedWindow(window_name, cv::WINDOW_FREERATIO);
  cv::imshow(window_name, top_view_img);
  cv::waitKey(0);
}

int main() { showLidarTopview(); }
