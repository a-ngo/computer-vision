# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


## Project Tasks - Performance Evaluation

### FP.1
- implemented in camFusion_Student.cpp line 232-279
- used a multimap to track the pairs of bounding box indices
- counted the keypoint correspondences for each box pair
- to find the best match between previous and current frames

### FP.2
- implemented in camFusion_Student.cpp line 214-230
- sorted the vectors of lidar points from the previous and current frame
- and used the median to calculate the TTC 

### FP.3
- implemented in camFusion_Student.cpp line 146-176
- associate the bounding boxes with the keypoints it contains
- by calculating the euclidean distances between all keypoints
- and only allowing a distance which is lower than a defined threshold
- therefore, the outliers can be found and removed

### FP.4
- implemented in camFusion_Student.cpp line 180-201
- like lidar ttc calculation, here I also used the median for the camera ttc calculation
- hereby, the distance ratio is used for the calculation which is similar from the 2D feature tracking project 

### FP.5
- the results are listed in [method_comparison](method_comparison.ods)
- I did not find any significant outlier in the lidar TTC calculation
- the lidar ttc values are all in the range 8-17s
- the only thing that might be not plausible is that around frame 4,5 the ttc increases to about 16s
- this might be due to the calculation of the ttc, because I used the median distance of the point cloud

### FP.6
- as mentioned before, the results are listed in [method_comparison](method_comparison.ods)
- hereby, I compared all possible detector and descriptor combinations and saved the ttc values of lidar, camera, and the absolute delta between both
- observations
  - especially the HARRIS and ORB detectors produces very poor TTC estimations compared to lidar ttc
  - whereas FAST and SIFT achieved very good estimations


- detector_type, descriptor_type, frame, ttc_camera, ttc_lidar, abs(ttc_camera-ttc_lidar)
- snippet of the results table below:

- TOP 10
BRISK	 ORB	17	9.54524	9.54658	0.0013377
FAST	 ORB	6	12.6684	12.6787	0.0103077
FAST	 ORB	7	11.9712	11.9844	0.0131715
FAST	 SIFT	3	14.0617	14.091	0.029332
BRISK	 FREAK	11	12.8407	12.8086	0.0321132
ORB	 SIFT	10	11.1389	11.1746	0.035773
FAST	 ORB	2	12.6574	12.6142	0.0431052
ORB	 SIFT	16	9.56045	9.51617	0.0442762
FAST	 ORB	8	13.1719	13.1241	0.0477812
SIFT	 SIFT	2	12.5597	12.6142	0.0545167
SIFT	 BRISK	13	10.0226	9.96439	0.0582309

- WORST 10
ORB	 BRISK	5	520.246	15.9082	504.338
ORB	 SIFT	5	502.13	15.9082	486.222
ORB	 SIFT	4	209.183	16.6894	192.494
ORB	 BRIEF	9	140.565	13.0241	127.541
HARRIS	 ORB	3	-80.8525	14.091	94.9435
HARRIS	 SIFT	10	-76.9648	11.1746	88.1394
HARRIS	 SIFT	2	80.7525	12.6142	68.1383
ORB	 BRISK	12	51.3351	8.95978	42.3754
ORB	 FREAK	14	39.1271	9.59863	29.5285
ORB	 ORB	18	36.3076	8.3988	27.9088
ORB	 BRIEF	5	43.498	15.9082	27.5898
