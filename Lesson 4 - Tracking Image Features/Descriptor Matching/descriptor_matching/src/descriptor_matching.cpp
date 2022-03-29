#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

void matchDescriptors(cv::Mat &img_source, cv::Mat &img_ref,
                      std::vector<cv::KeyPoint> &k_pts_source,
                      std::vector<cv::KeyPoint> &k_pts_ref,
                      cv::Mat &desc_source, cv::Mat &desc_ref,
                      std::vector<cv::DMatch> &matches,
                      std::string descriptor_type, std::string matcher_type,
                      std::string selector_type) {
  // configure matcher
  bool crossCheck = true;
  cv::Ptr<cv::DescriptorMatcher> matcher;

  if (matcher_type.compare("MAT_BF") == 0) {
    int normType = descriptor_type.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING
                                                              : cv::NORM_L2;
    matcher = cv::BFMatcher::create(normType, crossCheck);
    std::cout << "BF matching cross-check=" << crossCheck;
  } else if (matcher_type.compare("MAT_FLANN") == 0) {
    if (desc_source.type() !=
        CV_32F) { // OpenCV bug workaround : convert binary
                  // descriptors to floating point due to a
                  // bug in current OpenCV implementation
      desc_source.convertTo(desc_source, CV_32F);
      desc_ref.convertTo(desc_ref, CV_32F);
    }
    matcher = cv::FlannBasedMatcher::create();
    std::cout << "FLANN matching";
  }

  // perform matching task
  // nearest neighbor (best match)
  if (selector_type.compare("SEL_NN") == 0) {
    double t = static_cast<double>(cv::getTickCount());
    // Finds the best match for each descriptor in desc1
    matcher->match(desc_source, desc_ref, matches);
    t = static_cast<double>(cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " (NN) with n=" << matches.size() << " matches in "
              << 1000 * t / 1.0 << " ms" << std::endl;
  } else if (selector_type.compare("SEL_KNN") == 0) {
    // implement k-nearest-neighbor matching
    double t = static_cast<double>(cv::getTickCount());

    // k nearest neighbors (k=2) -> finds the two best matches for each
    // descriptor
    int k = 2;
    const double kThreshold = 0.8;
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch(desc_source, desc_ref, knn_matches, k);

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

    t = static_cast<double>(cv::getTickCount() - t) / cv::getTickFrequency();
    std::cout << " (NN) with n=" << matches.size() << " matches in "
              << 1000 * t / 1.0 << " ms" << std::endl;
  }

  // visualize results
  cv::Mat matchImg = img_ref.clone();
  cv::drawMatches(img_source, k_pts_source, img_ref, k_pts_ref, matches,
                  matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1),
                  std::vector<char>(),
                  cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  std::string windowName =
      "Matching keypoints between two camera images (best 50)";
  cv::namedWindow(windowName, 7);
  cv::imshow(windowName, matchImg);
  cv::waitKey(0);
}

int main() {
  cv::Mat img_source = cv::imread("../images/img1gray.png");
  cv::Mat img_ref = cv::imread("../images/img2gray.png");

  // choose descriptor set
  std::string data = "BRISK_large";
  // data = "BRISK_small";
  data = "SIFT";

  std::vector<cv::KeyPoint> kpts_source, kpts_ref;
  readKeypoints(std::string("../dat/C35A5_KptsSource_" + data + ".dat").c_str(),
                kpts_source);
  readKeypoints(std::string("../dat/C35A5_KptsRef_" + data + ".dat").c_str(),
                kpts_ref);

  cv::Mat desc_source, desc_ref;
  readDescriptors(
      std::string("../dat/C35A5_DescSource_" + data + ".dat").c_str(),
      desc_source);
  readDescriptors(std::string("../dat/C35A5_DescRef_" + data + ".dat").c_str(),
                  desc_ref);

  std::vector<cv::DMatch> matches;

  // choose method
  // MAT_BF or MAT_FLANN
  std::string matcher_type = "MAT_BF";
  matcher_type = "MAT_FLANN";

  // NORM_HAMMING or NORM_L2
  std::string descriptor_type = "DES_BINARY";

  // SEL_NN or SEL_KNN
  std::string selector_type = "SEL_KNN";
  matchDescriptors(img_source, img_ref, kpts_source, kpts_ref, desc_source,
                   desc_ref, matches, descriptor_type, matcher_type,
                   selector_type);
}
