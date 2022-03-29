#include <algorithm>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>

// these includes provide the data structures for managing 3D Lidar points and
// 2D keypoints
#include "dataStructures.h" // you do not need to look into this file
#include "structIO.hpp"     // you do not need to look into this file

// Purpose: Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images Notes:
// - please take a look at the main()-function first
// - kptsPrev and kptsCurr are the input keypoint sets, kptMatches are the
// matches between the two sets,
//   frameRate is required to compute the delta time between frames and TTC will
//   hold the result of the computation
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC) {
  // compute distance ratios between all matched keypoints
  std::vector<double> dist_ratios; // stores the distance ratios for all
                                   // keypoints between curr. and prev. frame
  for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1;
       ++it1) { // outer keypoint loop

    // get current keypoint and its matched partner in the prev. frame
    cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
    cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

    for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end();
         ++it2) { // inner keypoint loop

      double minDist = 100.0; // min. required distance

      // get next keypoint and its matched partner in the prev. frame
      cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
      cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

      // compute distances and distance ratios
      double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
      double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

      if (distPrev > std::numeric_limits<double>::epsilon() &&
          distCurr >= minDist) { // avoid division by zero

        double dist_ratio = distCurr / distPrev;
        dist_ratios.push_back(dist_ratio);
      }
    } // eof inner loop over all matched kpts
  }   // eof outer loop over all matched kpts

  // only continue if list of distance ratios is not empty
  if (dist_ratios.size() == 0) {
    TTC = NAN;
    return;
  }

  // compute camera-based TTC from distance ratios
  double meanDistRatio =
      std::accumulate(dist_ratios.begin(), dist_ratios.end(), 0.0) /
      dist_ratios.size();

  size_t n = dist_ratios.size() / 2;
  std::nth_element(dist_ratios.begin(), dist_ratios.begin() + n,
                   dist_ratios.end());
  double meadian_distance_ratio = dist_ratios[n];

  double dT = 1 / frameRate;
  // TTC = -dT / (1 - meanDistRatio);
  TTC = -dT / (1 - meadian_distance_ratio);
}

int main() {
  /* camera-based TTC calculation procedure
    1. detect bounding boxes of objects
    2. calculate key points/features within bboxes of objects
    3. match key points between frames to establish correspondences
    4. calculate relative distances between changes between these
       correspondences within the bounding box and compute stable estimate of
       TTC
  */

  // step 1: read pre-recorded keypoint sets from file
  // Note that the function "readKeypoints" is a helper function that is able to
  // read pre-saved results from disk so that you can focus on TTC computation
  // based on a defined set of keypoints and matches. The task you need to solve
  // in this example does not require you to look into the data structures.
  std::vector<cv::KeyPoint> kptsSource, kptsRef;
  readKeypoints(
      "../dat/C23A5_KptsSource_AKAZE.dat",
      kptsSource); // readKeypoints("./dat/C23A5_KptsSource_SHI-BRISK.dat"
  readKeypoints("../dat/C23A5_KptsRef_AKAZE.dat",
                kptsRef); // readKeypoints("./dat/C23A5_KptsRef_SHI-BRISK.dat"

  // step 2: read pre-recorded keypoint matches from file
  std::vector<cv::DMatch> matches;
  readKptMatches(
      "../dat/C23A5_KptMatches_AKAZE.dat",
      matches); // readKptMatches("./dat/C23A5_KptMatches_SHI-BRISK.dat",
                // matches);

  // step 3: compute the time-to-collision based on the pre-recorded data
  double ttc;
  computeTTCCamera(kptsSource, kptsRef, matches, 10.0, ttc);
  std::cout << "ttc = " << ttc << "s" << std::endl;
}
