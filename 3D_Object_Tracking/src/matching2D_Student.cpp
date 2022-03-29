
#include "matching2D.hpp"
#include <numeric>

using namespace std;

// Find best matches for keypoints in two camera images based on several
// matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource,
                      std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef, std::vector<cv::DMatch> &matches,
                      std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
  // configure matcher
  bool crossCheck = false;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  // matcher type
  if (matcherType.compare("MAT_BF") == 0) {
    int norm_type = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING
                                                              : cv::NORM_L2;

    matcher = cv::BFMatcher::create(norm_type, crossCheck);
  } else if (matcherType.compare("MAT_FLANN") == 0) {
    if (descSource.type() != CV_32F) { // OpenCV bug workaround : convert binary
                                       // descriptors to floating point due to a
                                       // bug in current OpenCV implementation
      descSource.convertTo(descSource, CV_32F);
      descRef.convertTo(descRef, CV_32F);
    }
    matcher = cv::FlannBasedMatcher::create();
  }

  // perform matching task
  // nearest neighbor (best match)
  if (selectorType.compare("SEL_NN") == 0) {
    // Finds the best match for each descriptor in desc1
    matcher->match(descSource, descRef, matches);
  } else if (selectorType.compare("SEL_KNN") == 0) {
    // k nearest neighbors (k=2)
    int k = 2;
    const double kThreshold = 0.8;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(descSource, descRef, knn_matches, k);

    // filter matches using descriptor distance ratio test
    for (size_t i = 0; i < knn_matches.size(); ++i) {
      // calculate ratio between descriptor distances to best and second - best
      // match
      double ratio = knn_matches[i][0].distance / knn_matches[i][1].distance;

      // if ratio higher then threshold -> ambiguous
      if (ratio < kThreshold) {
        // push back best fitted descriptor
        matches.push_back(knn_matches[i][0]);
      }
    }
  }
}

// Use one of several types of state-of-art descriptors to uniquely identify
// keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                   cv::Mat &descriptors, string descriptorType) {
  // select appropriate descriptor
  cv::Ptr<cv::DescriptorExtractor> extractor;
  if (descriptorType.compare("BRISK") == 0) {
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for
                               // sampling the neighbourhood of a keypoint.
    extractor = cv::BRISK::create(threshold, octaves, patternScale);
  } else if (descriptorType.compare("BRIEF") == 0) {
    extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();

  } else if (descriptorType.compare("ORB") == 0) {
    extractor = cv::ORB::create();

  } else if (descriptorType.compare("FREAK") == 0) {
    extractor = cv::xfeatures2d::FREAK::create();

  } else if (descriptorType.compare("AKAZE") == 0) {
    extractor = cv::AKAZE::create();

  } else if (descriptorType.compare("SIFT") == 0) {
    extractor = cv::SIFT::create();
  }

  // perform feature description
  double t = (double)cv::getTickCount();
  extractor->compute(img, keypoints, descriptors);
  t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
  // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0
  //      << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img,
                           bool bVis) {
  // compute detector parameters based on image size
  int blockSize = 4; //  size of an average block for computing a derivative
                     //  covariation matrix over each pixel neighborhood
  double maxOverlap = 0.0; // max. permissible overlap between two features in %
  double minDistance = (1.0 - maxOverlap) * blockSize;
  int maxCorners =
      img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

  double qualityLevel = 0.01; // minimal accepted quality of image corners
  double k = 0.04;

  // Apply corner detection
  double t = static_cast<double>(cv::getTickCount());
  vector<cv::Point2f> corners;
  cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
                          cv::Mat(), blockSize, false, k);

  // add corners to result vector
  for (auto it = corners.begin(); it != corners.end(); ++it) {

    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    keypoints.push_back(newKeyPoint);
  }
  t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
  // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in "
  //      << 1000 * t / 1.0 << " ms" << endl;

  // visualize results
  if (bVis) {
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "Shi-Tomasi Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
  }
}

void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_gray,
                        bool b_vis) {
  // Detector parameters
  int block_size = 2;
  int aperture_size = 3;
  int min_response = 100;
  double k = 0.04;
  // window size should be odd in order to center it on a pixel and have
  // symmetry in all directions
  int sliding_window_size = 7;
  int sw_dist = std::floor(sliding_window_size / 2);

  cv::Mat dst, dst_norm, dst_norm_scaled;
  dst = cv::Mat::zeros(img_gray.size(), CV_32FC1);
  cv::cornerHarris(img_gray, dst, block_size, aperture_size, k,
                   cv::BORDER_DEFAULT);

  cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
  cv::convertScaleAbs(dst_norm, dst_norm_scaled);

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
      }
    }
  }

  // visualize results
  if (b_vis) {
    cv::Mat vis_image = img_gray.clone();
    cv::drawKeypoints(img_gray, keypoints, vis_image, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string window_name = "Harris Corner Detector Results";
    cv::namedWindow(window_name, 1);
    imshow(window_name, vis_image);
    cv::waitKey(0);
  }
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img_gray,
                        std::string detector_type, bool b_vis) {
  cv::Ptr<cv::FeatureDetector> detector;

  // determine detector type
  if (detector_type.compare("FAST") == 0) {
    detector = cv::FastFeatureDetector::create();
  } else if (detector_type.compare("BRISK") == 0) {
    detector = cv::BRISK::create();
  } else if (detector_type.compare("ORB") == 0) {
    detector = cv::ORB::create();
  } else if (detector_type.compare("AKAZE") == 0) {
    detector = cv::AKAZE::create();
  } else if (detector_type.compare("SIFT") == 0) {
    detector = cv::SIFT::create();
  }

  detector->detect(img_gray, keypoints);

  // visualize results
  if (b_vis) {
    cv::Mat vis_image = img_gray.clone();
    cv::drawKeypoints(img_gray, keypoints, vis_image, cv::Scalar::all(-1),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    std::string window_name = "detector_type Detector Results";
    cv::namedWindow(window_name, 1);
    imshow(window_name, vis_image);
    cv::waitKey(0);
  }
}
