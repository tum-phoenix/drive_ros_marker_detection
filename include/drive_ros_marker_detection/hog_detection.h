#ifndef HOG_DETECTION_H
#define HOG_DETECTION_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect.hpp>

namespace drive_ros_marker_detection {

class HogDetection
{
public:
  HogDetection(ros::NodeHandle nh, ros::NodeHandle pnh);
  bool init();
private:
  void imageCallback(const sensor_msgs::ImageConstPtr &msg);
  void processImage(const cv::Mat &img_in);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  image_transport::Subscriber image_sub_;
  std::string internal_image_encoding_;
  cv::Size window_size_;
  bool use_preprocessing_;
  cv::HOGDescriptor hog_;
};

}

#endif
