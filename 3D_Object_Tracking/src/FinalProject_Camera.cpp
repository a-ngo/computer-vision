
/* INCLUDES FOR THIS PROJECT */
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

#include "camFusion.hpp"
#include "dataStructures.h"
#include "lidarData.hpp"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
  /* INIT VARIABLES AND DATA STRUCTURES */

  // data location
  std::string dataPath = "../";

  // camera
  std::string imgBasePath = dataPath + "images/";
  std::string imgPrefix =
      "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
  std::string imgFileType = ".png";
  int imgStartIndex = 0; // first file index to load (assumes Lidar and camera
                         // names have identical naming convention)
  int imgEndIndex = 18;  // last file index to load
  int imgStepWidth = 1;
  int imgFillWidth =
      4; // no. of digits which make up the file index (e.g. img-0001.png)

  // object detection
  std::string yoloBasePath = dataPath + "dat/yolo/";
  std::string yoloClassesFile = yoloBasePath + "coco.names";
  std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

  // Lidar
  std::string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
  std::string lidarFileType = ".bin";

  // calibration data for camera and lidar
  cv::Mat P_rect_00(
      3, 4,
      cv::DataType<double>::type); // 3x4 projection matrix after rectification
  cv::Mat R_rect_00(4, 4,
                    cv::DataType<double>::type); // 3x3 rectifying rotation to
                                                 // make image planes co-planar
  cv::Mat RT(
      4, 4,
      cv::DataType<double>::type); // rotation matrix and translation vector

  RT.at<double>(0, 0) = 7.533745e-03;
  RT.at<double>(0, 1) = -9.999714e-01;
  RT.at<double>(0, 2) = -6.166020e-04;
  RT.at<double>(0, 3) = -4.069766e-03;
  RT.at<double>(1, 0) = 1.480249e-02;
  RT.at<double>(1, 1) = 7.280733e-04;
  RT.at<double>(1, 2) = -9.998902e-01;
  RT.at<double>(1, 3) = -7.631618e-02;
  RT.at<double>(2, 0) = 9.998621e-01;
  RT.at<double>(2, 1) = 7.523790e-03;
  RT.at<double>(2, 2) = 1.480755e-02;
  RT.at<double>(2, 3) = -2.717806e-01;
  RT.at<double>(3, 0) = 0.0;
  RT.at<double>(3, 1) = 0.0;
  RT.at<double>(3, 2) = 0.0;
  RT.at<double>(3, 3) = 1.0;

  R_rect_00.at<double>(0, 0) = 9.999239e-01;
  R_rect_00.at<double>(0, 1) = 9.837760e-03;
  R_rect_00.at<double>(0, 2) = -7.445048e-03;
  R_rect_00.at<double>(0, 3) = 0.0;
  R_rect_00.at<double>(1, 0) = -9.869795e-03;
  R_rect_00.at<double>(1, 1) = 9.999421e-01;
  R_rect_00.at<double>(1, 2) = -4.278459e-03;
  R_rect_00.at<double>(1, 3) = 0.0;
  R_rect_00.at<double>(2, 0) = 7.402527e-03;
  R_rect_00.at<double>(2, 1) = 4.351614e-03;
  R_rect_00.at<double>(2, 2) = 9.999631e-01;
  R_rect_00.at<double>(2, 3) = 0.0;
  R_rect_00.at<double>(3, 0) = 0;
  R_rect_00.at<double>(3, 1) = 0;
  R_rect_00.at<double>(3, 2) = 0;
  R_rect_00.at<double>(3, 3) = 1;

  P_rect_00.at<double>(0, 0) = 7.215377e+02;
  P_rect_00.at<double>(0, 1) = 0.000000e+00;
  P_rect_00.at<double>(0, 2) = 6.095593e+02;
  P_rect_00.at<double>(0, 3) = 0.000000e+00;
  P_rect_00.at<double>(1, 0) = 0.000000e+00;
  P_rect_00.at<double>(1, 1) = 7.215377e+02;
  P_rect_00.at<double>(1, 2) = 1.728540e+02;
  P_rect_00.at<double>(1, 3) = 0.000000e+00;
  P_rect_00.at<double>(2, 0) = 0.000000e+00;
  P_rect_00.at<double>(2, 1) = 0.000000e+00;
  P_rect_00.at<double>(2, 2) = 1.000000e+00;
  P_rect_00.at<double>(2, 3) = 0.000000e+00;

  // misc
  double sensorFrameRate =
      10.0 / imgStepWidth; // frames per second for Lidar and camera
  int dataBufferSize = 2;  // no. of images which are held in memory (ring
                           // buffer) at the same time
  bool bVis = false;                 // visualize results

  /* MAIN LOOP OVER ALL IMAGES */
  std::vector<std::string> detector_types{"HARRIS", "FAST", "BRISK",
                                          "ORB",    "SIFT", "AKAZE"};
  std::vector<std::string> descriptor_types{"BRISK", "BRIEF", "ORB",
                                            "FREAK", "SIFT",  "AKAZE"};
  double ttc_lidar{0};
  double ttc_camera{0};

  for (const auto &detector_type : detector_types) {
    for (const auto &descriptor_type : descriptor_types) {

      // reset buffer
      std::vector<DataFrame> dataBuffer;

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

      for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex;
          imgIndex += imgStepWidth) {
        /* LOAD IMAGE INTO BUFFER */

        // reset metrics
        ttc_lidar = 0;
        ttc_camera = 0;

        // assemble filenames for current index
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth)
                  << imgStartIndex + imgIndex;
        std::string imgFullFilename =
            imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        // std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects((dataBuffer.end() - 1)->cameraImg,
                      (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                      nmsThreshold, yoloBasePath, yoloClassesFile,
                      yoloModelConfiguration, yoloModelWeights, bVis);

        // std::cout << "#2 : DETECT & CLASSIFY OBJECTS done" << std::endl;

        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        std::string lidarFullFilename =
            imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0,
              minR = 0.1; // focus on ego lane
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        // std::cout << "#3 : CROP LIDAR POINTS done" << std::endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor =
            0.10; // shrinks each bounding box by the given percentage to avoid 3D
                  // object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes,
                            (dataBuffer.end() - 1)->lidarPoints, shrinkFactor,
                            P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        bVis = true;
        if (bVis) {
          show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0),
                        cv::Size(2000, 2000), true);
        }
        bVis = false;

        // std::cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << std::endl;

        // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
        // continue; // skips directly to the next image without processing what
        // comes beneath

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray,
                    cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        std::vector<cv::KeyPoint> keypoints;

        if (detector_type.compare("SHITOMASI") == 0) {
          detKeypointsShiTomasi(keypoints, imgGray, bVis);
        } else if (detector_type.compare("HARRIS") == 0) {
          detKeypointsHarris(keypoints, imgGray, bVis);
        } else if (detector_type.compare("FAST") == 0 ||
                  detector_type.compare("BRISK") == 0 ||
                  detector_type.compare("ORB") == 0 ||
                  detector_type.compare("AKAZE") == 0 ||
                  detector_type.compare("SIFT") == 0) {
          detKeypointsModern(keypoints, imgGray, detector_type, bVis);
        } else {
          std::cerr << detector_type << " not supported!" << std::endl;
        }

        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts) {
          int maxKeypoints = 50;

          if (detector_type.compare("SHITOMASI") ==
              0) { // there is no response info, so keep the first 50 as they are
                  // sorted in descending quality order
            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
          }
          cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
          std::cout << " NOTE: Keypoints have been limited!" << std::endl;
        }

        // push keypoints and descriptor for current frame to end of data buffer
        (dataBuffer.end() - 1)->keypoints = keypoints;

        // std::cout << "#5 : DETECT KEYPOINTS done" << std::endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints,
                      (dataBuffer.end() - 1)->cameraImg, descriptors,
                      descriptor_type);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        // std::cout << "#6 : EXTRACT DESCRIPTORS done" << std::endl;

        // wait until at least two images have been processed
        if (dataBuffer.size() > 1) {

          /* MATCH KEYPOINT DESCRIPTORS */

          std::vector<cv::DMatch> matches;
          std::string matcherType = "MAT_BF"; // MAT_BF, MAT_FLANN
          std::string distance_type =
              (descriptor_type.compare("SIFT") == 0) ? "DES_HOG" : "DES_BINARY";
          std::string selectorType = "SEL_KNN"; // SEL_NN, SEL_KNN

          matchDescriptors((dataBuffer.end() - 2)->keypoints,
                          (dataBuffer.end() - 1)->keypoints,
                          (dataBuffer.end() - 2)->descriptors,
                          (dataBuffer.end() - 1)->descriptors, matches,
                          descriptor_type, matcherType, selectorType);

          // store matches in current data frame
          (dataBuffer.end() - 1)->kptMatches = matches;

          // std::cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

          /* TRACK 3D OBJECT BOUNDING BOXES */

          //// STUDENT ASSIGNMENT
          //// TASK FP.1 -> match list of 3D objects
          /// (vector<BoundingBox>) between
          /// current and previous frame (implement ->matchBoundingBoxes)
          std::map<int, int> bbBestMatches;
          // associate bounding boxes between current
          // and previous frame using keypoint matches
          matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2),
                            *(dataBuffer.end() - 1));
          //// EOF STUDENT ASSIGNMENT

          // store matches in current data frame
          (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

          // std::cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << std::endl;

          /* COMPUTE TTC ON OBJECT IN FRONT */

          // loop over all BB match pairs
          for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
              it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
            // find bounding boxes associates with current match
            BoundingBox *prevBB, *currBB;
            for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
              if (it1->second == it2->boxID) // check wether current match partner
                                            // corresponds to this BB
              {
                currBB = &(*it2);
              }
            }

            for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
              if (it1->first == it2->boxID) // check wether current match partner
                                            // corresponds to this BB
              {
                prevBB = &(*it2);
              }
            }

            // compute TTC for current match
            // only compute TTC if we have Lidar points
            if (currBB->lidarPoints.size() > 0 && prevBB->lidarPoints.size() > 0) {
              //// STUDENT ASSIGNMENT
              //// TASK FP.2 -> compute time-to-collision based on Lidar
              /// data (implement -> computeTTCLidar)
              double ttcLidar;
              computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints,
                              sensorFrameRate, ttcLidar);
              ttc_lidar = ttcLidar;
              //// EOF STUDENT ASSIGNMENT

              //// STUDENT ASSIGNMENT
              // TASK FP.3 -> assign enclosed keypoint matches to
              //                bounding box (implement -> clusterKptMatchesWithROI)
              // TASK FP.4 -> compute time-to-collision based on
              //                            camera (implement -> computeTTCCamera)
              double ttcCamera;
              clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints,
                                      (dataBuffer.end() - 1)->keypoints,
                                      (dataBuffer.end() - 1)->kptMatches);
              computeTTCCamera((dataBuffer.end() - 2)->keypoints,
                              (dataBuffer.end() - 1)->keypoints,
                              currBB->kptMatches, sensorFrameRate, ttcCamera);
              ttc_camera = ttcCamera;
              //// EOF STUDENT ASSIGNMENT

              bVis = true;
              if (bVis) {
                cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00,
                                    R_rect_00, RT, &visImg);
                cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                              cv::Point(currBB->roi.x + currBB->roi.width,
                                        currBB->roi.y + currBB->roi.height),
                              cv::Scalar(0, 255, 0), 2);

                char str[200];
                sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar,
                        ttcCamera);
                putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2,
                        cv::Scalar(0, 0, 255));

                std::string windowName = "Final Results : TTC";
                cv::namedWindow(windowName, 4);
                cv::imshow(windowName, visImg);
                // std::cout << "Press key to continue to next frame" << std::endl;
                cv::waitKey(0);
              }
              bVis = false;              
            } // eof TTC computation
          }   // eof loop over all BB matches
        }

        // print results for each combination for each frame
        std::cout << detector_type << ", "
                  << descriptor_type << ", "
                  << imgIndex << ", "
                  << ttc_camera << ", "
                  << ttc_lidar << ", " 
                  << std::abs(ttc_lidar - ttc_camera) << std::endl;
      } // eof loop over all images
    }
  }

  return 0;
}
