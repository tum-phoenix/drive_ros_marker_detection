#include <ros/ros.h>
#include <nodelet/nodelet.h>
#include <image_transport/image_transport.h>
#include <memory>
#include <unordered_map>
#include <opencv2/core/core.hpp>

namespace drive_ros_marker_detection {

struct ModelType
{
  ModelType() : level_(), rotation_() {}
  ModelType(int level, double rotation) : level_(level), rotation_(rotation) {}
  bool operator==(const ModelType& other) const {return (level_ == other.level_ &&
                                              rotation_ == other.rotation_);}
  bool operator<(const ModelType& other) const
  {
    if (level_ == other.level_)
      return rotation_ < other.rotation_;
    else
      return level_ < other.level_;
  }
  int level_;
  double rotation_;
};

struct ModelIdentifier
{
  ModelIdentifier(): model_name_(), model_type_() {}
  ModelIdentifier(const std::string &model_name, const ModelType &model_type): model_name_(model_name), model_type_(model_type)
  {}

  std::string model_name_;
  ModelType model_type_;
};

class MarkerDetection
{
public:
  MarkerDetection(ros::NodeHandle nh, ros::NodeHandle pnh);

  bool init();

  void imageCallback(const sensor_msgs::ImageConstPtr& msg);

private:
  std::pair<bool, double> matchModel(int i, int j, const ModelIdentifier &model_id, const cv::Mat &pyramid_grad_x,
                                     const cv::Mat &pyramid_grad_y, const cv::Mat &pyramid_magnitude);

  void drawTemplate(cv::Mat &img,
                    const std::pair<cv::Point2d, double> &template_coordinates,
                    const ModelIdentifier &model_id,
                    const cv::Vec3b &color = cv::Vec3b(0, 255, 0));

  void drawTextLabel(cv::Mat &img, const ModelType &m_type, bool success);

  void showTemplateAndLabel(cv::Mat &img, const std::pair<cv::Point2d, double> &template_coordinates,
                            const ModelIdentifier &model_id, bool success,
                            const std::string &window_name = "Currently checked template",
                            const cv::Vec3b &color = cv::Vec3b(0, 255, 0));

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::map<std::string, int> num_pyramid_levels_;
  int max_pyramid_level_;
  std::map<std::string, double> rotation_steps_lowest_level_;
  float greedy_factor_;
  float min_score_;
  bool display_results_;
  image_transport::Subscriber img_sub_;

  // edge models previously created using feature_extractor_node
  // map keys are the edge orientations of the templates, ordered in ascending pyramid order (low to high resolution)
  std::map<std::string, std::map<ModelType, cv::Mat> > magnitude_models_mat_;
  std::map<std::string, std::map<ModelType, cv::Mat> > grad_x_models_mat_;
  std::map<std::string, std::map<ModelType, cv::Mat> > grad_y_models_mat_;

  std::map<std::string, std::map<ModelType, std::vector<int> > > model_coordinates_x_;
  std::map<std::string, std::map<ModelType, std::vector<int> > > model_coordinates_y_;
  std::map<std::string, std::map<ModelType, std::vector<double> > > grad_x_models_;
  std::map<std::string, std::map<ModelType, std::vector<double> > > grad_y_models_;
  std::map<std::string, std::map<ModelType, std::vector<double> > > magnitude_models_;

  std::pair<double, double> search_bounds_rows_;
  std::pair<double, double> search_bounds_cols_;
  std::vector<std::string> model_names_;
};

class MarkerDetectionNodelet : public nodelet::Nodelet {
public:
    virtual void onInit();
private:
    std::unique_ptr<MarkerDetection> marker_detection_;
};

}
