#include <drive_ros_marker_detection/marker_detection.h>
#include <cv_bridge/cv_bridge.h>
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>

drive_ros_marker_detection::MarkerDetection::MarkerDetection(ros::NodeHandle nh, ros::NodeHandle pnh) :
  nh_(nh), pnh_(pnh), greedy_factor_(0.1), min_score_(2.0), display_results_(false), model_coordinates_x_(),
  model_coordinates_y_(), magnitude_models_(), grad_x_models_(), grad_y_models_(), magnitude_models_mat_(),
  grad_x_models_mat_(), grad_y_models_mat_(), img_sub_(), model_names_(), rotation_steps_lowest_level_(),
  num_pyramid_levels_(), max_pyramid_level_(), search_bounds_cols_(), search_bounds_rows_()
{
}

bool drive_ros_marker_detection::MarkerDetection::init()
{
  if (!pnh_.getParam("greedy_factor", greedy_factor_)) {
    ROS_ERROR_STREAM("Unable to get 'greedy_factor' parameter, shutting down!");
    return false;
  }

  if (!pnh_.getParam("min_score", min_score_)) {
    ROS_ERROR_STREAM("Unable to get 'min_score' parameter, shutting down!");
    return false;
  }

  if (!pnh_.getParam("display_results", display_results_)) {
    ROS_ERROR_STREAM("Unable to get 'display_results' parameter, shutting down!");
    return false;
  }

  std::vector<double> search_region_row;
  if (!pnh_.getParam("search_region_row", search_region_row)) {
    ROS_INFO_STREAM("Unable to get 'search_region_row' parameter, using the full row range for search");
    search_bounds_rows_.first = 0.0;
    search_bounds_rows_.second = 1.0;
  }
  else
  {
    CV_Assert(search_region_row.size() == 2);
    search_bounds_rows_.first = search_region_row[0];
    search_bounds_rows_.second = search_region_row[1];
  }

  std::vector<double> search_region_col;
  if (!pnh_.getParam("search_region_col", search_region_col)) {
    ROS_INFO_STREAM("Unable to get 'search_region_col' parameter, using the column range for search");
    search_bounds_cols_.first = 0.0;
    search_bounds_cols_.second = 1.0;
  }
  else
  {
    CV_Assert(search_region_col.size() == 2);
    search_bounds_cols_.first = search_region_col[0];
    search_bounds_cols_.second = search_region_col[1];
  }


  bool cv_mat_stored;
  if (!pnh_.getParam("cv_mat_stored", cv_mat_stored)) {
    ROS_ERROR_STREAM("Unable to get 'cv_mat_stored' parameter, shutting down!");
    return false;
  }
  image_transport::ImageTransport it(nh_);
  // todo: different callbacks depending on how the edge models are stored
  if (cv_mat_stored)
    img_sub_ = it.subscribe("img_in", 1, &drive_ros_marker_detection::MarkerDetection::imageCallback, this);
  else
    img_sub_ = it.subscribe("img_in", 1, &drive_ros_marker_detection::MarkerDetection::imageCallback, this);

  std::vector<std::string> model_paths;
  if (!pnh_.getParam("model_path", model_paths)) {
    ROS_ERROR_STREAM("Unable to get 'model_path' parameter, shutting down!");
    return false;
  }
  if (model_paths.size() == 0)
  {
    ROS_ERROR_STREAM("No saved point models provided in 'model_path' parameter!");
    return false;
  }

  // load yaml files containing models stored as cv::Mat
  cv::FileStorage fs;
  std::string model_name;
  int pyramid_level, max_pyramid_level = -1;
  double rotation;
  ModelType m_type;
  std::string rotation_str;
  std::vector<double> temp_vec_vals;
  std::vector<int> temp_vec_coords;

  for (const std::string& model_path : model_paths)
  {
    fs = cv::FileStorage(model_path, cv::FileStorage::READ);
    model_name = (std::string)fs["model_name"];
    num_pyramid_levels_[model_name] = (int)fs["num_pyramid_levels"];
    if (num_pyramid_levels_[model_name] > max_pyramid_level)
      max_pyramid_level = num_pyramid_levels_[model_name];
    rotation_steps_lowest_level_[model_name] = (double)fs["rotation_step_lowest_level"];
    model_names_.push_back(model_name);

    ROS_DEBUG_STREAM("[marker_detection] After checking path "<<model_path<<" found model: "<<model_name
                     <<" with pyramid levels: "<<num_pyramid_levels_[model_name]<<" and rotation step at lowest: "
                     <<rotation_steps_lowest_level_[model_name]);

    cv::FileNode model = fs["model"];
    for(cv::FileNodeIterator pyramid_it = model.begin(); pyramid_it != model.end(); ++pyramid_it)
    {
      // todo: remove prepended string pyramid_level_
      pyramid_level = std::stoi(std::string((*pyramid_it).name()).erase(0, std::string("pyramid_level_").size()));
      m_type.level_ = pyramid_level;
      for (cv::FileNodeIterator rotation_it = (*pyramid_it).begin(); rotation_it != (*pyramid_it).end();
           ++rotation_it)
      {
        // todo: remove prepended string rotation_
        rotation_str = std::string((*rotation_it).name()).erase(0, std::string("rotation_").size());
        std::replace(rotation_str.begin(), rotation_str.end(), '_', '.');
        rotation = std::stod(rotation_str);
        m_type.rotation_ = rotation;
        if (cv_mat_stored)
        {
          fs["grad_x_mat"] >> grad_x_models_mat_[model_name][m_type];
          fs["grad_y_mat"] >> grad_y_models_mat_[model_name][m_type];
          fs["magnitudes_mat"] >> magnitude_models_mat_[model_name][m_type];
        }
        else
        {
          (*rotation_it)["coordinates_x"] >> temp_vec_coords;
          model_coordinates_x_[model_name][m_type] = temp_vec_coords;
          temp_vec_coords.clear();
          (*rotation_it)["coordinates_y"] >> temp_vec_coords;
          model_coordinates_y_[model_name][m_type] = temp_vec_coords;
          temp_vec_coords.clear();
          (*rotation_it)["grad_x"] >> temp_vec_vals;
          grad_x_models_[model_name][m_type] = temp_vec_vals;
          temp_vec_vals.clear();
          (*rotation_it)["grad_y"] >> temp_vec_vals;
          grad_y_models_[model_name][m_type] = temp_vec_vals;
          temp_vec_vals.clear();
          (*rotation_it)["magnitudes"] >> temp_vec_vals;
          magnitude_models_[model_name][m_type] = temp_vec_vals;
          temp_vec_vals.clear();

          CV_Assert(model_coordinates_x_[model_name][m_type].size() == grad_x_models_[model_name][m_type].size() &&
                    grad_x_models_[model_name][m_type].size() == grad_y_models_[model_name][m_type].size() &&
                    grad_y_models_[model_name][m_type].size() == magnitude_models_[model_name][m_type].size() &&
                    magnitude_models_[model_name][m_type].size() == model_coordinates_y_[model_name][m_type].size());
        }
      }
    }
    fs.release();
  }
  max_pyramid_level_ = max_pyramid_level;
  fs.release();
  return true;
}

void drive_ros_marker_detection::MarkerDetection::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv_bridge::CvImageConstPtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvShare(msg);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  // go up to highest pyramid level for full search
  cv::Mat search_img = cv_ptr->image.clone();
  std::vector<cv::Mat> pyramid_images(max_pyramid_level_);
  pyramid_images[max_pyramid_level_ - 1] = search_img.clone();
  std::vector<cv::Mat> pyramid_grad_x;
  std::vector<cv::Mat> pyramid_grad_y;
  std::vector<cv::Mat> pyramid_magnitudes;

  if (display_results_)
  {
    cv::namedWindow("Pyramid image", CV_WINDOW_NORMAL);
  }
  for (int i = 1; i < max_pyramid_level_; ++i)
  {
    cv::pyrDown(search_img, search_img);
    pyramid_images[max_pyramid_level_ - i - 1] = search_img.clone();
    if (display_results_)
    {
      cv::imshow("Pyramid image", pyramid_images[max_pyramid_level_ - i - 1]);
    }
  }

  // edge magnitude of highest level for full search
  cv::Mat grad_x = cv::Mat::zeros(search_img.rows, search_img.cols, CV_64FC1);
  cv::Mat grad_y = cv::Mat::zeros(search_img.rows, search_img.cols, CV_64FC1);
  cv::Sobel(search_img*1/255, grad_x, CV_64F, 1, 0, 3, 0.125);  // find X derivatives
  cv::Sobel(search_img*1/255, grad_y, CV_64F, 0, 1, 3, 0.125);  // find Y derivatives
  pyramid_grad_x.push_back(grad_x.clone());
  pyramid_grad_y.push_back(grad_y.clone());
  if (display_results_)
  {
    cv::namedWindow("Sobel x", CV_WINDOW_NORMAL);
    cv::namedWindow("Sobel y", CV_WINDOW_NORMAL);
    cv::imshow("Sobel x", grad_x);
    cv::imshow("Sobel y", grad_y);
  }

  // todo: calculate edge magnitude matrix for the incoming image
  cv::Mat source_magnitude = cv::Mat::zeros(search_img.rows, search_img.cols, CV_64FC1);
  cv::magnitude(grad_x, grad_y, source_magnitude);
  if (display_results_)
  {
    cv::namedWindow("Edge magnitude", CV_WINDOW_NORMAL);
    cv::imshow("Edge magnitude", source_magnitude);
    cv::waitKey(0);
  }
  pyramid_magnitudes.push_back(source_magnitude.clone());

  CV_Assert(grad_x.size == grad_y.size);
  CV_Assert(grad_x.type() == grad_y.type());

  std::vector<std::pair<cv::Point2d, double> > candidate_detections; // position in image (relative to full width/height)/rotation
  // full search at highest level for each model
  ModelType m_type(0, 0.0);

  double rotation_step;
  int num_pyramid_levels;
  double partial_score;
  std::pair<bool, double> matching_result;

  for (const std::string& model_name : model_names_)
  {
    num_pyramid_levels = num_pyramid_levels_[model_name];
    rotation_step = rotation_steps_lowest_level_[model_name]*std::pow(2.0, num_pyramid_levels);

    for( int i = search_bounds_rows_.first * grad_x.rows; i < search_bounds_cols_.second * grad_x.rows; i++ )
    {
      for( int j = search_bounds_cols_.first * grad_x.cols; j < search_bounds_cols_.second * grad_x.cols; j++ )
      {
        for (double rotation = 0.0; rotation < 360.0; rotation += rotation_step)
        {
          ROS_INFO_STREAM("checking model "<<model_name<<" with rotation "<<rotation<<" at image coordinates "<<i<<", "<<j);
          m_type.rotation_ = rotation;
          if (matchModel(i, j, m_type, pyramid_grad_x[0], pyramid_grad_y[0], pyramid_magnitudes[0], model_name).first)
          {
            ROS_INFO_STREAM("Candidate detection found at: "<<i<<", "<<j<<" with rotation "<<rotation<<" saved to candidate: "<<cv::Point2d(i/grad_x.rows, j/grad_x.cols));
            candidate_detections.push_back(std::pair<cv::Point2d, double>(cv::Point2d((double)i/(double)grad_x.rows,
                                                                                      (double)j/(double)grad_x.cols),
                                                                          rotation));
          }
          else {
            ROS_INFO_STREAM("Candidate detection not found!");
          }
        }
      }
    }

    // now: propagate down in the pyramid to confirm detections
    for( int step = 1; step < num_pyramid_levels; step++ )
    {
      cv::Mat temp_mat = cv::Mat::zeros(pyramid_images[step].rows, pyramid_images[step].cols, CV_64FC1);
      if (step >= pyramid_grad_x.size())
      {
        cv::Sobel(pyramid_images[step], temp_mat, CV_64F, 1, 0, 3, 0.125);
        pyramid_grad_x.push_back(temp_mat.clone());
        cv::Sobel(pyramid_images[step], temp_mat, CV_64F, 0, 1, 3, 0.125);
        pyramid_grad_y.push_back(temp_mat.clone());
        cv::magnitude(pyramid_grad_x.back(), pyramid_grad_y.back(), temp_mat);
        pyramid_magnitudes.push_back(temp_mat.clone());
      }

      grad_x = pyramid_grad_x[step];
      grad_y = pyramid_grad_y[step];
      source_magnitude = pyramid_magnitudes[step];
      m_type.level_ -= 1;
      rotation_step *= 0.5;
      search_img = pyramid_images[step];

      int cand_x, cand_y;

      for (int cand_idx=0; cand_idx < candidate_detections.size(); ++cand_idx)
      {
        std::pair<cv::Point2d, int> candidate_detection = candidate_detections[cand_idx];
        cand_x = (int) (candidate_detection.first.x*search_img.rows);
        cand_y = (int) (candidate_detection.first.y*search_img.cols);

        std::pair<cv::Point2d, int> best_position;
        double best_score = 0.0;
        ROS_INFO_STREAM("Candidate detection: "<<candidate_detection.first);

        for (int i = cand_x -1; i <= cand_x + 1; ++i)
        {
          for (int j = cand_y - 1; j <= cand_y + 1; ++ j)
          {
            for (double rotation = candidate_detection.second - rotation_step;
                 rotation <= candidate_detection.second + rotation_step; rotation += rotation_step)
            {
              ROS_INFO_STREAM("checking candidate for model "<<model_name<<" saved as "<<candidate_detection.first<<" with rotation "<<rotation<<" at image coordinates "<<i<<", "<<j<<" for pyramid level "<<step);
              m_type.rotation_ = rotation;
              matching_result = matchModel(i, j, m_type, pyramid_grad_x[step], pyramid_grad_y[step],
                                           pyramid_magnitudes[step], model_name);
              if (matching_result.first)
              {
                if (matching_result.second > best_score)
                {
                  ROS_INFO_STREAM("Candidate detection confirmed at: "<<i<<", "<<j<<" with rotation "<<rotation);
                  best_score = partial_score;
                  best_position = std::pair<cv::Point2d, double>(cv::Point2d((double)i/(double)search_img.rows,
                                                                             (double)j/(double)search_img.cols),
                                                              rotation);
                }
              }
              else
              {
                ROS_WARN_STREAM("Lost track of candidate detection "<<candidate_detection.first<<" with rotation: "<<candidate_detection.second<<" at pyramid step "<<step);
              }
            }
          }
        }

        // if all child detections have failed we have lost the track down the pyramid
        if (best_score > 0.0)
        {
          candidate_detections[cand_idx] = best_position;
        }
        else
        {
          candidate_detections.erase(candidate_detections.begin() + cand_idx);
        }
      }
      rotation_step *= 0.5;
    }

    // draw detections
    if (display_results_)
    {
      std::stringstream ss;
      ss << "Detections of " << model_name << " in image";
      cv::Mat display_image;
      display_image = pyramid_images[num_pyramid_levels-1].clone();
      m_type.level_ = 0;
      for (const std::pair<cv::Point2d, int>& detection : candidate_detections)
      {
        ROS_INFO_STREAM("Found the model "<<model_name<<" at: "<<detection.first.x<<", "<<detection.first.y<<" and rotation: "<<detection.second);
        m_type.rotation_ = detection.second;
        drawTemplate(display_image, detection, model_coordinates_x_[model_name][m_type],
                     model_coordinates_y_[model_name][m_type], ss.str());
      }
    }
  }
}

std::pair<bool, double> drive_ros_marker_detection::MarkerDetection::matchModel(int i, int j,
                                                                                const ModelType m_type,
                                                                                const cv::Mat &pyramid_grad_x,
                                                                                const cv::Mat &pyramid_grad_y,
                                                                                const cv::Mat &pyramid_magnitude,
                                                                                const std::string &model_name)
{
  double partial_sum = 0.0;
  double partial_score = 0.0;

  int num_coordinates = model_coordinates_x_[model_name][m_type].size();
  ROS_INFO_STREAM("Number of coordinates to check: "<<num_coordinates);
  for( int m = 0; m < num_coordinates; m++)
  {
    int cur_x    = i + model_coordinates_x_[model_name][m_type][m];    // template X coordinate
    int cur_y    = j + model_coordinates_y_[model_name][m_type][m];    // template Y coordinate
    double i_tx    = grad_x_models_[model_name][m_type][m];    // template X derivative
    double i_ty    = grad_y_models_[model_name][m_type][m];    // template Y derivative

    if(cur_x < 0 || cur_y < 0|| cur_x > pyramid_grad_x.rows-1 || cur_y > pyramid_grad_x.cols-1)
      continue;

    double i_sx = pyramid_grad_x.at<double>(cur_x, cur_y); // get corresponding  X derivative from source image
    double i_sy = pyramid_grad_y.at<double>(cur_x, cur_y); // get corresponding  Y derivative from source image

    if((i_sx != 0 || i_sy != 0) && ( i_tx!=0 || i_ty!=0))
    {
      //partial Sum  = Sum of(((Source X derivative* Template X drivative)
      //+ Source Y derivative * Template Y derivative)) / Edge
      //magnitude of(Template)* edge magnitude of(Source))
      partial_sum = partial_sum + ((i_sx*i_tx)+(i_sy*i_ty)) /
          (magnitude_models_[model_name][m_type][m] * pyramid_magnitude.at<double>(cur_x, cur_y));

    }

    // stoping criterias to search for model
    double norm_min_score = min_score_ / num_coordinates; // precompute minumum score
    double norm_greediness = ((1- greedy_factor_ * min_score_)/(1 - greedy_factor_))
        / num_coordinates; // precompute greedniness

    int sum_coords = m + 1;
    partial_score = partial_sum / sum_coords;
    // check termination criteria
    // if partial score score is less than the score than
    // needed to make the required score at that position
    // break serching at that coordinate.
    if( partial_score < (MIN((min_score_ - 1) +
                             norm_greediness*sum_coords, norm_min_score * sum_coords)))
    {
      ROS_INFO_STREAM("[marker_detection] MatchModel(): Did "<<m<<" checks before cancel");
      return std::pair<bool, double>(false, partial_score);
    }
//            ROS_INFO_STREAM("Partial score after "<<m<<" checks: "<<partial_score<<" cancel conditions: "<< (min_score_ - 1) + norm_greediness*sum_coords<<" and "<<norm_min_score * sum_coords);
  }

  return std::pair<bool, double>(true, partial_score);
}

void drive_ros_marker_detection::MarkerDetection::drawTemplate(cv::Mat &img,
                                                               const std::pair<cv::Point2d, double> &template_coordinates,
                                                               const std::vector<int> &pixel_coordinates_x,
                                                               const std::vector<int> &pixel_coordinates_y,
                                                               const std::string &window_name,
                                                               const cv::Vec3b &color)
{
  if (img.type() == CV_8UC1)
  {
    cv::cvtColor(img, img, CV_GRAY2BGR);
  }
  else if (img.type() != CV_8UC3)
  {
    ROS_WARN_STREAM("[marker_detection] Invalid image type provived to drawTemplate(): "<<img.type()<<" supported "
                                                                                                      "types are: "
                    <<CV_8UC1<<" and "<<CV_8UC3);
    return;
  }

  cv::namedWindow(window_name, CV_WINDOW_NORMAL);
  cv::Point image_anchor((int)(img.rows*template_coordinates.first.x), (int)(img.cols*template_coordinates.first.y));

  for (int i=0; i<pixel_coordinates_x.size(); ++i)
  {
    img.at<cv::Vec3b>(cv::Point(pixel_coordinates_x[i], pixel_coordinates_y[i])+image_anchor) = color;
  }

  cv::imshow(window_name, img);
  cv::waitKey(0);
  return;
}

void drive_ros_marker_detection::MarkerDetectionNodelet::onInit()
{
  marker_detection_.reset(new MarkerDetection(getNodeHandle(), getPrivateNodeHandle()));
  if (!marker_detection_->init()) {
    ROS_ERROR("MarkerDetectionNodelet failed to initialize!");
    // nodelet failing will kill the entire loader anyway
    ros::shutdown();
  }
  else {
    ROS_INFO("MarkerDetection nodelet succesfully initialized");
  }
}

PLUGINLIB_EXPORT_CLASS(drive_ros_marker_detection::MarkerDetectionNodelet, nodelet::Nodelet)
