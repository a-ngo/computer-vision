/* INCLUDES FOR THIS PROJECT */
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <sstream>
#include <vector>

#include "dataStructures.h"
#include "matching2D.hpp"

template <typename T1, size_t T2> size_t count(const std::array<T1, T2> vec) {
  size_t sum{0};
  for (const auto &element : vec) {
    sum += element;
  }
  return sum;
}

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */

  // data location
  std::string data_path = "../";

  // camera
  std::string img_base_path = data_path + "images/";
  std::string img_prefix =
      "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
  std::string img_file_type = ".png";
  int img_start_index = 0; // first file index to load (assumes Lidar and camera
                           // names have identical naming convention)
  int img_end_index = 9;   // last file index to load
  int img_fill_width =
      4; // no. of digits which make up the file index (e.g. img-0001.png)

  // misc
  int data_buffer_size = 2; // no. of images which are held in memory (ring
                            // buffer) at the same time

  bool b_vis = false; // visualize results

  size_t preceding_vehicle_keypoints{0};
  size_t matches_number{0};
  double keypoint_detection_time{0};
  double descriptor_extraction_time{0};

  std::vector<std::string> detector_types{"HARRIS", "FAST", "BRISK",
                                          "ORB",    "SIFT", "AKAZE"};
  std::vector<std::string> descriptor_types{"BRISK", "BRIEF", "ORB",
                                            "FREAK", "SIFT",  "AKAZE"};

  for (const auto &detector_type : detector_types) {
    for (const auto &descriptor_type : descriptor_types) {
      // skip incompatible detector and descriptor combinations
      if (detector_type.compare("AKAZE") == 0 &&
              descriptor_type.compare("AKAZE") != 0 ||
          detector_type.compare("AKAZE") != 0 &&
              descriptor_type.compare("AKAZE") == 0) {
        continue;
      }

      if (detector_type.compare("SIFT") == 0 &&
          descriptor_type.compare("ORB") == 0) {
        continue;
      }

      // reset metrics and buffer
      preceding_vehicle_keypoints = 0;
      matches_number = 0;
      keypoint_detection_time = 0;
      descriptor_extraction_time = 0;
      std::vector<DataFrame> data_buffer;

      std::cout << "Detection/descriptor combi: ";
      std::cout << detector_type << " " << descriptor_type << std::endl;

      for (size_t img_index = 0; img_index <= img_end_index - img_start_index;
           img_index++) {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        std::ostringstream img_number;
        img_number << std::setfill('0') << std::setw(img_fill_width)
                   << img_start_index + img_index;
        std::string img_full_filename =
            img_base_path + img_prefix + img_number.str() + img_file_type;

        // load image from file and convert to grayscale
        cv::Mat img, img_gray;
        img = cv::imread(img_full_filename);
        cv::cvtColor(img, img_gray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT
        //// TASK MP.1 -> replace the following code with ring buffer
        ///               of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img_gray;

        if (data_buffer.size() > data_buffer_size) {
          data_buffer.erase(data_buffer.begin());
        }
        data_buffer.push_back(frame);

        //// EOF STUDENT ASSIGNMENT
        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        // create empty feature list for current image
        std::vector<cv::KeyPoint> keypoints;

        //// STUDENT ASSIGNMENT
        //// TASK MP.2 -> add the following keypoint detectors in file
        /// matching2D.cpp and enable string-based selection based on
        /// detectorType
        /// /
        ///-> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
        double t = static_cast<double>(cv::getTickCount());

        if (detector_type.compare("SHITOMASI") == 0) {
          detKeypointsShiTomasi(keypoints, img_gray, b_vis);
        } else if (detector_type.compare("HARRIS") == 0) {
          detKeypointsHarris(keypoints, img_gray, b_vis);
        } else if (detector_type.compare("FAST") == 0 ||
                   detector_type.compare("BRISK") == 0 ||
                   detector_type.compare("ORB") == 0 ||
                   detector_type.compare("AKAZE") == 0 ||
                   detector_type.compare("SIFT") == 0) {
          detKeypointsModern(keypoints, img_gray, detector_type, b_vis);
        } else {
          std::cerr << detector_type << " not supported!" << std::endl;
        }

        t = static_cast<double>(cv::getTickCount() - t) /
            cv::getTickFrequency();
        keypoint_detection_time += t;

        //// EOF STUDENT ASSIGNMENT

        //// STUDENT ASSIGNMENT
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle

        // only keep keypoints on the preceding vehicle
        std::vector<size_t> keypoint_indices_outside_rect;

        bool b_focus_on_vehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (b_focus_on_vehicle) {
          // find outlier
          for (size_t keypoint_ind = 0; keypoint_ind < keypoints.size();
               ++keypoint_ind) {
            if (!keypoints.at(keypoint_ind).pt.inside(vehicleRect)) {
              keypoint_indices_outside_rect.push_back(keypoint_ind);
            }
          }

          // remove outlier
          for (size_t idx = keypoint_indices_outside_rect.size(); idx-- > 0;) {
            keypoints.at(idx) = keypoints.back();
            keypoints.pop_back();
          }

          preceding_vehicle_keypoints += keypoints.size();
        }

        //// EOF STUDENT ASSIGNMENT

        // optional : limit number of keypoints (helpful for debugging and
        // learning)
        bool b_limit_kpts = false;
        if (b_limit_kpts) {
          int max_keypoints = 20;

          if (detector_type.compare("SHITOMASI") ==
              0) { // there is no response info, so keep the first 50 as they
                   // are sorted in descending quality order
            keypoints.erase(keypoints.begin() + max_keypoints, keypoints.end());
          }
          cv::KeyPointsFilter::retainBest(keypoints, max_keypoints);
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (data_buffer.end() - 1)->keypoints = keypoints;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT
        //// TASK MP.4 -> add the following descriptors in file
        /// matching2D.cpp and
        /// enable string-based selection based on descriptorType / -> BRIEF,
        /// ORB, FREAK, AKAZE, SIFT

        // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        cv::Mat descriptors;
        t = static_cast<double>(cv::getTickCount());
        descKeypoints((data_buffer.end() - 1)->keypoints,
                      (data_buffer.end() - 1)->cameraImg, descriptors,
                      descriptor_type);
        t = static_cast<double>(cv::getTickCount() - t) /
            cv::getTickFrequency();
        descriptor_extraction_time += t;
        //// EOF STUDENT ASSIGNMENT

        // push descriptors for current frame to end of data buffer
        (data_buffer.end() - 1)->descriptors = descriptors;

        // wait until at least two images have been processed
        if (data_buffer.size() > 1) {
          /* MATCH KEYPOINT DESCRIPTORS */
          std::vector<cv::DMatch> matches;
          std::string matcher_type = "MAT_BF"; // MAT_BF, MAT_FLANN
          std::string distance_type =
              (descriptor_type.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY";
          std::string selector_type = "SEL_KNN"; // SEL_NN, SEL_KNN

          //// STUDENT ASSIGNMENT
          //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
          //// TASK MP.6 -> add KNN match selection and perform
          /// descriptor distance ratio filtering with t=0.8

          t = static_cast<double>(cv::getTickCount());
          matchDescriptors((data_buffer.end() - 2)->keypoints,
                           (data_buffer.end() - 1)->keypoints,
                           (data_buffer.end() - 2)->descriptors,
                           (data_buffer.end() - 1)->descriptors, matches,
                           distance_type, matcher_type, selector_type);
          t = static_cast<double>(cv::getTickCount() - t) /
              cv::getTickFrequency();
          //// EOF STUDENT ASSIGNMENT

          // store matches in current data frame
          (data_buffer.end() - 1)->kptMatches = matches;

          matches_number += matches.size();

          // visualize matches between current and previous image
          // b_vis = true;
          if (b_vis) {
            cv::Mat match_img = ((data_buffer.end() - 1)->cameraImg).clone();
            cv::drawMatches((data_buffer.end() - 2)->cameraImg,
                            (data_buffer.end() - 2)->keypoints,
                            (data_buffer.end() - 1)->cameraImg,
                            (data_buffer.end() - 1)->keypoints, matches,
                            match_img, cv::Scalar::all(-1), cv::Scalar::all(-1),
                            std::vector<char>(),
                            cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string window_name =
                "Matching keypoints between two camera images";
            cv::namedWindow(window_name, 7);
            cv::imshow(window_name, match_img);
            cv::waitKey(0);
          }
          b_vis = false;
        }
      }
      std::cout << "Keypoints: " << preceding_vehicle_keypoints << std::endl;
      std::cout << "Matched keypoints: " << matches_number << std::endl;
      std::cout << "Time needed: " << std::setprecision(3)
                << keypoint_detection_time + descriptor_extraction_time << "\n"
                << std::endl;
    }
  }

  return 0;
}
