#include <drive_ros_marker_detection/hog_detection.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>

namespace drive_ros_marker_detection {

HogDetection::HogDetection(ros::NodeHandle nh, ros::NodeHandle pnh) :
  nh_(nh),
  pnh_(pnh),
  window_size_(64, 128),
  internal_image_encoding_("8UC1"),
  hog_(window_size_, cv::Size(16, 16), cv::Size(8, 8), cv::Size(8, 8), 9),
  use_preprocessing_(false)
{

}

bool HogDetection::init()
{
//  std::string svm_save_path;
//  if (!pnh_.getParam("svm_save_path", svm_save_path))
//  {
//    ROS_ERROR("Failed to get svm_save_path parameter, shutting down!");
//    return false;
//  }
//  hog_.load(svm_save_path);

  // for benchmarking, just load a default people detector (window size has to be 64x128)
  hog_.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

  image_transport::ImageTransport it(pnh_);
  image_sub_ = it.subscribe("img_in", 1, &HogDetection::imageCallback, this);
  return true;
}

void HogDetection::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImageConstPtr img_in = cv_bridge::toCvShare(msg, internal_image_encoding_);
  processImage(img_in->image);
}

void HogDetection::processImage(const cv::Mat &img_in)
{
  // connected component detection used as preprocessing (can be used to drastically reduce the number of SVM calls)
  if (use_preprocessing_)
  {
    cv::Mat thresholded_image;
    cv::threshold(img_in, thresholded_image, 200.0, 255.0, cv::THRESH_BINARY);
    cv::namedWindow("thresholded image", cv::WINDOW_NORMAL);
    cv::imshow("thresholded image", thresholded_image);

    // use k-means to get clusters
    cv::Mat label_image(thresholded_image.size(), CV_32S);
    cv::Mat stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(thresholded_image, label_image, stats, centroids, 8, CV_32S);

    // get bounding rectangle to preprocess based on size and aspect ratio
    int x, y, w, h;
    for(int i=0; i<stats.rows; i++) {
      x = stats.at<int>(cv::Point(0, i));
      y = stats.at<int>(cv::Point(1, i));
      w = stats.at<int>(cv::Point(2, i));
      h = stats.at<int>(cv::Point(3, i));
    }

    std::vector<cv::Vec3b> colors(nLabels);
    colors[0] = cv::Vec3b(0, 0, 0);//background
    for(int label = 1; label < nLabels; ++label)
    {
      colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
    }
    cv::Mat draw_image(label_image.size(), CV_8UC3);
    for(int r = 0; r < draw_image.rows; ++r)
    {
      for(int c = 0; c < draw_image.cols; ++c)
      {
        int label = label_image.at<int>(r, c);
        cv::Vec3b &pixel = draw_image.at<cv::Vec3b>(r, c);
        pixel = colors[label];
      }
    }

    std::vector<cv::Mat> candidate_hog_features;
    cv::Mat hog_features_out;
    for(int label = 1; label < nLabels; ++label)
    {
      x = stats.at<int>(cv::Point(0, label));
      y = stats.at<int>(cv::Point(1, label));
      w = stats.at<int>(cv::Point(2, label));
      h = stats.at<int>(cv::Point(3, label));

      if (w > 20 && w < 80 && h > 20 && h < 80 && std::abs(std::max(w/h, h/w)-1.0) < 0.8)
      {
        cv::rectangle(draw_image, cv::Point(x, y), cv::Point(x+w, y+h), colors[label]);
        candidate_hog_features.push_back(hog_features_out.clone());
      }
    }
    cv::namedWindow("Connected components", cv::WINDOW_NORMAL);
    cv::imshow("Connected components", draw_image);
    cv::waitKey(0);
  }
  else
  {
    std::vector<cv::Point> detections;
    hog_.detect(img_in, detections);

    cv::Mat color_img;
    cv::cvtColor(img_in, color_img, CV_GRAY2BGR);
    cv::namedWindow("Detections in image", CV_WINDOW_NORMAL);

    for (const cv::Point& detection: detections)
    {
      cv::rectangle(color_img, cv::Rect(cv::Point(detection.x - window_size_.width*0.5,
                                                  detection.y - window_size_.height*0.5), window_size_),
                    cv::Scalar(255, 0, 0));
    }
    cv::imshow("Detections in image", color_img);
    cv::waitKey(10);
  }
}

}

