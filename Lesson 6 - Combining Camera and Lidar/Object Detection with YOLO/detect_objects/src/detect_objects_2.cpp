#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "dataStructures.h"

void detectObjects2() {
  // load image from file
  // cv::Mat img = cv::imread("../images/s_thrun.jpg");
  // cv::Mat img = cv::imread("../images/0000000000.png");
  cv::Mat img = cv::imread("../images/times_square.jpg");

  // load class names from file
  std::string yoloBasePath = "../dat/yolo/";
  std::string yoloClassesFile = yoloBasePath + "coco.names";
  std::string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
  std::string yoloModelWeights = yoloBasePath + "yolov3.weights";

  std::vector<std::string> classes;
  std::ifstream ifs(yoloClassesFile.c_str());
  std::string line;
  while (getline(ifs, line))
    classes.push_back(line);

  // load neural network
  cv::dnn::Net net =
      cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
  net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);

  // with gpu set DNN_TARGET_OPENCL
  net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

  // generate 4D blob from input image to feed to network
  cv::Mat blob;
  double scalefactor = 1 / 255.0;
  cv::Size size = cv::Size(416, 416);
  cv::Scalar mean = cv::Scalar(0, 0, 0);
  bool swapRB = false;
  bool crop = false;
  cv::dnn::blobFromImage(img, blob, scalefactor, size, mean, swapRB, crop);

  // Get names of output layers
  std::vector<cv::String> names;
  // get indices of output layers, i.e. layers with unconnected outputs
  std::vector<int> outLayers = net.getUnconnectedOutLayers();
  // get names of all layers in the network
  std::vector<cv::String> layersNames = net.getLayerNames();

  // Get the names of the output layers in names
  names.resize(outLayers.size());
  for (size_t i = 0; i < outLayers.size(); ++i) {
    names[i] = layersNames[outLayers[i] - 1];
  }

  // invoke forward propagation through network
  std::vector<cv::Mat> netOutput;
  net.setInput(blob);
  net.forward(netOutput, names);

  // Scan through all bounding boxes and keep only the ones with high confidence
  const float conf_threshold = 0.20;
  std::vector<int> classIds;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  for (size_t i = 0; i < netOutput.size(); ++i) {
    float *data = (float *)netOutput[i].data;
    for (int j = 0; j < netOutput[i].rows; ++j, data += netOutput[i].cols) {
      cv::Mat scores = netOutput[i].row(j).colRange(5, netOutput[i].cols);
      cv::Point classId;
      double confidence;

      // Get the value and location of the maximum score
      cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
      if (confidence > conf_threshold) {
        cv::Rect box;
        int cx, cy;
        cx = static_cast<int>(data[0] * img.cols);
        cy = static_cast<int>(data[1] * img.rows);
        box.width = static_cast<int>(data[2] * img.cols);
        box.height = static_cast<int>(data[3] * img.rows);
        // left
        box.x = cx - box.width / 2;
        // top
        box.y = cy - box.height / 2;

        boxes.push_back(box);
        classIds.push_back(classId.x);
        confidences.push_back(static_cast<float>(confidence));
      }
    }
  }

  // perform non-maxima suppression
  // Non-maximum suppression threshold
  const float nms_threshold = 0.4;
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
  std::vector<BoundingBox> bounding_boxes;
  for (auto it = indices.begin(); it != indices.end(); ++it) {
    BoundingBox bounding_box;
    bounding_box.roi = boxes[*it];
    bounding_box.classID = classIds[*it];
    bounding_box.confidence = confidences[*it];
    // zero-based unique identifier for this bounding box
    bounding_box.boxID = static_cast<int>(bounding_boxes.size());

    bounding_boxes.push_back(bounding_box);
  }

  // show results
  cv::Mat vis_img = img.clone();
  for (auto it = bounding_boxes.begin(); it != bounding_boxes.end(); ++it) {
    // Draw rectangle displaying the bounding box
    int top, left, width, height;
    top = (*it).roi.y;
    left = (*it).roi.x;
    width = (*it).roi.width;
    height = (*it).roi.height;
    cv::rectangle(vis_img, cv::Point(left, top),
                  cv::Point(left + width, top + height), cv::Scalar(0, 255, 0),
                  2);

    std::string label = cv::format("%.2f", (*it).confidence);
    label = classes[((*it).classID)] + ":" + label;

    // Display label at the top of the bounding box
    int baseLine;
    cv::Size label_size =
        getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
    top = std::max(top, label_size.height);
    rectangle(vis_img, cv::Point(left, top - round(1.5 * label_size.height)),
              cv::Point(left + round(1.5 * label_size.width), top + baseLine),
              cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(vis_img, label, cv::Point(left, top), cv::FONT_ITALIC, 0.75,
                cv::Scalar(0, 0, 0), 1);
  }

  std::string windowName = "Object classification";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, vis_img);
  cv::waitKey(0);
}

int main() { detectObjects2(); }
