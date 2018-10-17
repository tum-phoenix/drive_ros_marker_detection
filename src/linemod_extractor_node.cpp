#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem/path.hpp>
#include <algorithm>
#include <drive_ros_marker_detection/linemod.hpp>

void drawResponse(const std::vector<cv::linemod::Template>& templates,
                  int num_modalities, cv::Mat& dst, cv::Point offset, int T)
{
  static const cv::Scalar COLORS[5] = { CV_RGB(0, 0, 255),
                                        CV_RGB(0, 255, 0),
                                        CV_RGB(255, 255, 0),
                                        CV_RGB(255, 140, 0),
                                        CV_RGB(255, 0, 0) };

  for (int m = 0; m < num_modalities; ++m)
  {
    // NOTE: Original demo recalculated max response for each feature in the TxT
    // box around it and chose the display color based on that response. Here
    // the display color just depends on the modality.
    cv::Scalar color = COLORS[m];

    for (int i = 0; i < (int)templates[m].features.size(); ++i)
    {
      cv::linemod::Feature f = templates[m].features[i];
      cv::Point pt(f.x + offset.x, f.y + offset.y);
      cv::circle(dst, pt, T / 2, color);
    }
  }
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "feature_extractor_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

#ifndef NDEBUG
  // give GDB time to attach
  ros::Duration(2.0).sleep();
#endif

  bool display_results;
  if (!pnh.getParam("display_results", display_results)) {
    ROS_ERROR_STREAM("Unable to get 'display_results' parameter, shutting down!");
    return -1;
  }

  std::vector<std::string> reference_image_paths;
  if (!pnh.getParam("reference_image_paths", reference_image_paths)) {
    ROS_ERROR_STREAM("Unable to get 'reference_image_paths' parameter, shutting down!");
    return -1;
  }

  std::string save_path, model_name;
  if (!pnh.getParam("save_path", save_path)) {
    ROS_ERROR_STREAM("Unable to get 'save_path' parameter, shutting down!");
    return -1;
  }

  cv::Ptr<cv::linemod::Detector> det = cv::linemod::getDefaultLINE();
  for (const std::string& reference_image_path: reference_image_paths)
  {
    cv::Mat reference_image = cv::imread(reference_image_path);

    if(!reference_image.data )
    {
      ROS_INFO("Could not open or find the reference image located at: %s", reference_image_path.c_str());
      return -1;
    }

    boost::filesystem::path path(reference_image_path);
    model_name = path.stem().string();

    std::vector<cv::Mat> model_sources;
    model_sources.push_back(reference_image);
    cv::Mat model_mask;
    cv::bitwise_not(reference_image, model_mask);
    det->addTemplate(model_sources, model_name, model_mask);
  }

  bool test_detector = false;
  if (!pnh.getParam("test_detector", test_detector))
    return -1;

  // test the detector by applying to video stills and drawing the results
  if (test_detector)
  {
    // WARN: only images whose size is dividable by 16 work (see: https://github.com/opencv/opencv/issues/4593)
    std::vector<std::string> test_image_paths;

    if (!pnh.getParam("test_image_paths", test_image_paths))
    {
      ROS_ERROR_STREAM("Unable to get 'test_image_paths' parameter, shutting down!");
      return -1;
    }

    std::vector<cv::Mat> test_images;
    for (const std::string& test_image_path : test_image_paths)
    {
      cv::Mat test_image = cv::imread(test_image_path);
      if(!test_image.data )
      {
        ROS_INFO("Could not open or find the image located at: %s", test_image_path.c_str());
        continue;
      }
      test_images.push_back(test_image.clone());
    }

    float test_threshold = 0.01;
    if (!pnh.getParam("test_threshold", test_threshold))
    {
      ROS_WARN_STREAM("Unable to get 'test_threshold' parameter, using default: "<<test_threshold);
      return -1;
    }

    cv::namedWindow("Detections in image", CV_WINDOW_NORMAL);
    std::vector<cv::linemod::Match> test_matches;
    for (const cv::Mat& test_image : test_images)
    {
      std::vector<cv::Mat> test_modalities;
      // todo: fill test modalities, default linemod uses only the image itself
      test_modalities.push_back(test_image.clone());
      det->match(test_modalities, test_threshold, test_matches);

      cv::Mat display = test_image.clone();

      // draw test matches
      for (int i = 0; i < test_matches.size(); ++i)
      {
        cv::linemod::Match m = test_matches[i];

        ROS_INFO("Similarity: %5.1f%%; x: %3d; y: %3d; class: %s; template: %3d\n",
                 m.similarity, m.x, m.y, m.class_id.c_str(), m.template_id);

        // Draw matching template
        const std::vector<cv::linemod::Template>& templates = det->getTemplates(m.class_id, m.template_id);
        drawResponse(templates, det->getModalities().size(), display, cv::Point(m.x, m.y), det->getT(0));
      }
      cv::imshow("Detections in image", display);
      cv::waitKey(0);
    }
  }

  return 0;
}
