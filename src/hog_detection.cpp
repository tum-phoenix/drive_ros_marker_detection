#include <drive_ros_marker_detection/hog_detection.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

namespace drive_ros_marker_detection {

HogDetection::HogDetection(ros::NodeHandle nh, ros::NodeHandle pnh) : nh_(nh), pnh_(pnh)
{

}

bool HogDetection::init()
{
  image_transport::ImageTransport it(pnh_);
  image_sub_ = it.subscribe("img_in", 1, &HogDetection::imageCallback, this);
  return true;
}

void HogDetection::imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImageConstPtr img_in = cv_bridge::toCvShare(msg);
  processImage(img_in->image);
}

void HogDetection::processImage(const cv::Mat &img_in)
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
  for(int i=0; i<stats.rows; i++) {
    int x = stats.at<int>(cv::Point(0, i));
    int y = stats.at<int>(cv::Point(1, i));
    int w = stats.at<int>(cv::Point(2, i));
    int h = stats.at<int>(cv::Point(3, i));
  }

  std::vector<cv::Vec3b> colors(nLabels);
  colors[0] = cv::Vec3b(0, 0, 0);//background
  for(int label = 1; label < nLabels; ++label){
      colors[label] = cv::Vec3b( (rand()&255), (rand()&255), (rand()&255) );
  }
  cv::Mat draw_image(label_image.size(), CV_8UC3);
  for(int r = 0; r < draw_image.rows; ++r){
      for(int c = 0; c < draw_image.cols; ++c){
          int label = label_image.at<int>(r, c);
          cv::Vec3b &pixel = draw_image.at<cv::Vec3b>(r, c);
          pixel = colors[label];
       }
   }
  cv::namedWindow("Connected components", cv::WINDOW_NORMAL);
  cv::imshow("Connected components", draw_image);
  cv::waitKey(0);

//  // Set up the blob detector
//  cv::SimpleBlobDetector::Params params = cv::SimpleBlobDetector::Params();
//  params.minThreshold = 200;
//  params.maxThreshold = 255;
//  params.thresholdStep = 10;
//  params.minArea = 100;
//  params.maxArea = 500;
//  params.minDistBetweenBlobs = 10;
//  params.filterByArea = true;
//  params.filterByCircularity = false;
//  params.filterByColor = false;
//  params.filterByConvexity = false;
//  params.filterByInertia = false;
//  cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create(params);

//  // Detect blobs.
//  std::vector<cv::KeyPoint> keypoints;
//  detector->detect(img_in, keypoints);

//  // Draw detected blobs as red circles.
//  // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
//  cv::Mat img_with_keypoints;
//  cv::drawKeypoints(img_in, keypoints, img_with_keypoints, cv::Scalar(0,0,255),
//                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS );

//  cv::namedWindow("image keypoints", cv::WINDOW_NORMAL);
//  cv::imshow("image keypoints", img_with_keypoints);
//  cv::waitKey(0);
}

}