#include <drive_ros_marker_detection/hog_detection.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "marker_detection");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

#ifndef NDEBUG
  // give GDB time to attach
  ros::Duration(2.0).sleep();
#endif

  drive_ros_marker_detection::HogDetection hog_detection(nh, pnh);
  if (!hog_detection.init()) {
    return 1;
  }
  else {
    ROS_INFO("HOG detection node succesfully initialized");
  }

  while (ros::ok()) {
    ros::spin();
  }
  return 0;
}
