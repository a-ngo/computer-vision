#ifndef dataStructures_h
#define dataStructures_h

#include <vector>

#include <opencv2/core.hpp>

struct DataFrame { // represents the available sensor information at the same
                   // time instance

  cv::Mat cameraImg;

  // 2D keypoints within camera image
  std::vector<cv::KeyPoint> keypoints;
  // keypoint descriptors
  cv::Mat descriptors;
  // keypoint matches between previous and current frame
  std::vector<cv::DMatch> kptMatches;
};

#endif /* dataStructures_h */
