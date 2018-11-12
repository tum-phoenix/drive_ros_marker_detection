#ifndef HOG_DETECTION_H
#define HOG_DETECTION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/ml.hpp"

namespace drive_ros_marker_detection {

class HogDetection
{
public:
  HogDetection(ros::NodeHandle nh, ros::NodeHandle pnh);
  bool init();
private:
  void imageCallback(const sensor_msgs::ImageConstPtr &msg);
  void processImage(const cv::Mat &img_in);

  void computeHOG(const cv::Size win_size, const cv::Mat &img_in, cv::Mat &gradient_out, bool use_flip);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::Subscriber image_sub_;
  cv::Ptr<cv::ml::SVM> svm_;
};

}

#endif
