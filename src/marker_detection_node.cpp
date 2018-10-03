#include <drive_ros_marker_detection/marker_detection.h>
#include <ros/ros.h>
#include <image_transport/image_transport.h>

int main(int argc, char** argv)
{
  ros::init(argc, argv, "marker_detection");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

#ifndef NDEBUG
  // give GDB time to attach
  ros::Duration(2.0).sleep();
#endif

  drive_ros_marker_detection::MarkerDetection marker_detection(nh, pnh);
  if (!marker_detection.init()) {
    return 1;
  }
  else {
    ROS_INFO("Marker detection node succesfully initialized");
  }

  while (ros::ok()) {
    ros::spin();
  }
  return 0;
}
