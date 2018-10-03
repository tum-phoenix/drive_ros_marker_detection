#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem/path.hpp>
#include <algorithm>

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

  std::string reference_image_path;
  if (!pnh.getParam("reference_image_path", reference_image_path)) {
    ROS_ERROR_STREAM("Unable to get 'reference_image_path' parameter, shutting down!");
    return -1;
  }
  cv::Mat reference_image = cv::imread(reference_image_path);

  if(!reference_image.data )
  {
    ROS_INFO("Could not open or find the image located at: %s", reference_image_path.c_str());
    return -1;
  }

  std::vector<int> model_template_size;
  if (!pnh.getParam("model_template_size", model_template_size)) {
    ROS_ERROR_STREAM("Unable to get 'model_template_size' parameter, shutting down!");
    return -1;
  }

  int num_pyramid_levels;
  if (!pnh.getParam("num_pyramid_levels", num_pyramid_levels)) {
    ROS_ERROR_STREAM("Unable to get 'num_pyramid_levels' parameter, shutting down!");
    return -1;
  }

  double rotation_step;
  if (!pnh.getParam("rotation_step", rotation_step)) {
    ROS_ERROR_STREAM("Unable to get 'rotation_step' parameter, shutting down!");
    return -1;
  }

  std::string save_path, model_name;
  if (!pnh.getParam("save_path", save_path)) {
    ROS_ERROR_STREAM("Unable to get 'save_path' parameter, shutting down!");
    return -1;
  }
  boost::filesystem::path path(save_path);
  model_name = path.stem().string();
  ROS_INFO_STREAM("BOOST FILENAME: "<<model_name);

  int background_color = 0, border_mode;
  if (!pnh.getParam("background_color", background_color)) {
    ROS_INFO_STREAM("Unable to get 'backround_color' parameter, using cv::BORDER_REPLICATE instead!");
    border_mode = cv::BORDER_REPLICATE;
  }
  else
  {
    border_mode = cv::BORDER_CONSTANT;
  }

  cv::resize(reference_image, reference_image, cv::Size(model_template_size[0], model_template_size[1]));

  cv::Mat converted_reference_image = cv::Mat::zeros(reference_image.rows, reference_image.cols, CV_8UC1);
  if (reference_image.channels() == 3)
    cv::cvtColor(reference_image, converted_reference_image, CV_RGB2GRAY);
  else
    converted_reference_image = reference_image;

  double rotation_rad;
  cv::Point image_center;
  cv::Mat rot_mat;
  cv::Mat rotated_reference_image;
  std::stringstream ss;

  ss.str("");
  ss << save_path << ".yaml";
  ROS_INFO_STREAM("Saving model to "<<ss.str());
  cv::FileStorage fs(ss.str(), cv::FileStorage::WRITE);
  fs << "model_name" << model_name;
  fs << "rotation_step_lowest_level" << rotation_step;
  fs << "num_pyramid_levels" << num_pyramid_levels;
  fs << "model";

  ss.str("");
  ss << save_path << "_mat.yaml";
  cv::FileStorage fs_mat(ss.str(), cv::FileStorage::WRITE);
  fs_mat << "model_name" << model_name;
  fs_mat << "rotation_step_lowest_level" << rotation_step;
  fs_mat << "num_pyramid_levels" << num_pyramid_levels;
  fs_mat << "model";

  fs << "{";
  fs_mat << "{";
  // generate edge model for each pyramid level
  for (int pyramid_level = 0; pyramid_level < num_pyramid_levels; ++pyramid_level)
  {
    ss.str("");
    ss << "pyramid_level_" << pyramid_level;
    ROS_INFO_STREAM("pyramid str "<< ss.str());
    fs << ss.str() << "{";
    fs_mat << ss.str() << "{";

    ROS_INFO_STREAM("Image size at pyramid level "<<pyramid_level<<": "<<converted_reference_image.size());

    for (double rotation = 0.0; rotation < 360.0; rotation += rotation_step)
    {

      if (rotation != 0.0)
      {
        // rotate around the image corner, makes computation easier - OpenCV doesn't have cropped rotation
        image_center = cv::Point((int)(converted_reference_image.rows/2), (int)(converted_reference_image.cols/2));
        rotation_rad = (double)(rotation)/180.0*M_PI;
        rot_mat = cv::getRotationMatrix2D(cv::Point(0, 0), rotation, 1.0);
        // rotate rectangle to calculate the extends of the image region
        cv::RotatedRect image_rotated(cv::Point(0, 0), converted_reference_image.size(), rotation);
        cv::Vec2d im_center_vec(converted_reference_image.rows*0.5, converted_reference_image.cols*0.5);
        cv::transpose(rot_mat, rot_mat);
        cv::Mat rotated_center_mat = rot_mat*cv::Mat(im_center_vec);
        cv::transpose(rot_mat, rot_mat);
        rot_mat.at<double>(0, 2) = image_rotated.boundingRect().width*0.5 - rotated_center_mat.at<double>(1, 0);
        rot_mat.at<double>(1, 2) = image_rotated.boundingRect().height*0.5 - rotated_center_mat.at<double>(0, 0);
        cv::warpAffine(converted_reference_image, rotated_reference_image, rot_mat,
                       cv::Size(image_rotated.boundingRect().width, image_rotated.boundingRect().height),
                       cv::INTER_LINEAR, border_mode, cv::Scalar(background_color));
      }
      else
      {
        rotated_reference_image = converted_reference_image.clone();
      }

      if (rotation != 0 && display_results)
      {
        cv::namedWindow("Rotated reference image", CV_WINDOW_NORMAL);
        cv::imshow("Rotated reference image", rotated_reference_image);
      }

      // step 1: detect edges in the image and group them by direction
      cv::Mat grad_x = cv::Mat::zeros(rotated_reference_image.rows, rotated_reference_image.cols, CV_64FC1);
      cv::Mat grad_y = cv::Mat::zeros(rotated_reference_image.rows, rotated_reference_image.cols, CV_64FC1);
      cv::Sobel( rotated_reference_image*1/255, grad_x, CV_64F, 1, 0/*, 3, 0.125*/); //gradient in X direction
      cv::Sobel( rotated_reference_image*1/255, grad_y, CV_64F, 0, 1/*, 3, 0.125*/); //gradient in Y direction

      CV_Assert(grad_x.size == grad_y.size);
      CV_Assert(grad_x.type() == grad_y.type());

      double magn = 0.0, direction = 0.0;
      cv::Mat magn_mat = cv::Mat::zeros(grad_x.rows, grad_x.cols, grad_x.type());
      cv::Mat orientations = magn_mat.clone();

      if (display_results)
      {
        cv::namedWindow("Sobel x", CV_WINDOW_NORMAL);
        cv::imshow("Sobel x", grad_x);
        cv::namedWindow("Sobel y", CV_WINDOW_NORMAL);
        cv::imshow("Sobel y", grad_y);
      }

      //Magnitude = Sqrt(gx^2 +gy^2)
      cv::magnitude(grad_x, grad_y, magn_mat);
      double min_gradient = 0.0, max_gradient = 0.0;
      cv::minMaxLoc(magn_mat, &min_gradient, &max_gradient);

      for(int i = 1; i < grad_x.rows; ++i )
      {
        const double* gx_ptr = grad_x.ptr<double>(i);
        const double* gy_ptr = grad_y.ptr<double>(i);
        for(int j = 1; j < grad_x.cols; ++j )
        {
          // read x, y derivatives
          double dx = gx_ptr[j];
          double dy = gy_ptr[j];

          //Direction = atan (Gy / Gx)
          direction = cv::fastAtan2((float)dy,(float)dx);

          // get closest angle from 0, 45, 90, 135 set
          if ( (direction>0 && direction < 22.5) ||
               (direction >157.5 && direction < 202.5) ||
               (direction>337.5 && direction<360)  )
            direction = 0;
          else if ( (direction>22.5 && direction < 67.5) ||
                    (direction >202.5 && direction <247.5)  )
            direction = 45;
          else if ( (direction >67.5 && direction < 112.5)||
                    (direction>247.5 && direction<292.5) )
            direction = 90;
          else if ( (direction >112.5 && direction < 157.5)||
                    (direction>292.5 && direction<337.5) )
            direction = 135;
          else
            direction = 0;

          orientations.at<double>(i, j) = (double)direction;
        }
      }

      if (display_results)
      {
        cv::namedWindow("Edge magnitude", CV_WINDOW_NORMAL);
        cv::imshow("Edge magnitude", magn_mat);
        cv::namedWindow("Edge direction", CV_WINDOW_NORMAL);
        cv::imshow("Edge direction", orientations);
      }

      // step 2: non-maximum suppression
      cv::Mat nms_edges = cv::Mat::zeros(grad_x.rows, grad_x.cols, grad_x.type());
      double left_pixel, right_pixel;
      for(int i = 1; i < grad_x.rows; i++ )
      {
        for(int j = 1; j < grad_x.cols; j++ )
        {
          switch ( orientations.at<uchar>(i, j) )
          {
          case 0:
            left_pixel = magn_mat.at<double>(i, j-1);
            right_pixel = magn_mat.at<double>(i, j+1);
            break;
          case 45:
            left_pixel  = magn_mat.at<double>(i-1, j+1);
            right_pixel = magn_mat.at<double>(i+1, j-1);
            break;
          case 90:
            left_pixel  =  magn_mat.at<double>(i-1, j);
            right_pixel =  magn_mat.at<double>(i+1, j);
            break;
          case 135:
            left_pixel  =  magn_mat.at<double>(i-1, j-1);
            right_pixel =  magn_mat.at<double>(i+1, j+1);
            break;
          }
          // compare current pixels value with adjacent pixels
          if ((  magn_mat.at<double>(i, j) < left_pixel ) || (magn_mat.at<double>(i, j) < right_pixel ) )
            nms_edges.at<double>(i, j) = 0.0;
          else
            nms_edges.at<double>(i, j) = magn_mat.at<double>(i, j)/max_gradient;
        }
      }

      if (display_results)
      {
        cv::namedWindow("Edge magnitude after NMS", CV_WINDOW_NORMAL);
        cv::imshow("Edge magnitude after NMS", nms_edges);
        cv::waitKey(0);
      }

      // step 3: hysteresis thresholding
      double min_contrast, max_contrast;
      if (!pnh.getParam("min_contrast", min_contrast)) {
        ROS_ERROR_STREAM("Unable to get 'min_contrast' parameter, shutting down!");
        return -1;
      }

      if (!pnh.getParam("max_contrast", max_contrast)) {
        ROS_ERROR_STREAM("Unable to get 'max_contrast' parameter, shutting down!");
        return -1;
      }
      // threshold below the min_contrast as those pixels are certainly not edges
      cv::Mat hys_edges;
      cv::threshold(nms_edges, hys_edges, min_contrast, -1, cv::THRESH_TOZERO);

      std::vector<int> template_coordinates_x;
      std::vector<int> template_coordinates_y;
      std::vector<double> template_magnitudes;
      std::vector<double> template_grad_x;
      std::vector<double> template_grad_y;
      for(int i = 1; i < grad_x.rows; i++ )
      {
        for(int j = 1; j < grad_x.cols; j++ )
        {
          if (hys_edges.at<double>(i, j) < max_contrast)
          {
            // if any of 8 neighboring pixel is not greater than max contraxt remove from edge
            if( (hys_edges.at<double>(i-1, j-1) < max_contrast) &&
                (hys_edges.at<double>(i-1, j) < max_contrast)   &&
                (hys_edges.at<double>(i-1, j+1) < max_contrast) &&
                (hys_edges.at<double>(i, j-1) < max_contrast)     &&
                (hys_edges.at<double>(i, j+1) < max_contrast) &&
                (hys_edges.at<double>(i+1, j-1) < max_contrast) &&
                (hys_edges.at<double>(i+1, j) < max_contrast)   &&
                (hys_edges.at<double>(i+1, j+1) < max_contrast))
            {
              hys_edges.at<double>(i, j) = 0.0;
            }
            else if (hys_edges.at<double>(i, j) != 0.0)
            {
              template_coordinates_x.push_back(i);
              template_coordinates_y.push_back(j);
              template_magnitudes.push_back(hys_edges.at<double>(i, j));
            }
          }
          else
          {
            template_coordinates_x.push_back(i);
            template_coordinates_y.push_back(j);
            template_magnitudes.push_back(hys_edges.at<double>(i, j));
          }
        }
      }

      for (int i=0; i<template_coordinates_x.size(); ++i)
      {
        template_grad_x.push_back(grad_x.at<double>(template_coordinates_x[i], template_coordinates_y[i]));
        template_grad_y.push_back(grad_y.at<double>(template_coordinates_x[i], template_coordinates_y[i]));
      }

      // step 4: display and save the resulting model
      if (display_results)
      {
        cv::namedWindow("Final edge magnitude after hysteresis thresholding", CV_WINDOW_NORMAL);
        cv::imshow("Final edge magnitude after hysteresis thresholding", hys_edges);
        cv::waitKey(0);
      }

      // save edge magnitude
      // convert back to 8-bit to save
      cv::Mat model_edge_magnitude = cv::Mat::zeros(hys_edges.rows, hys_edges.cols, CV_8UC1);
      hys_edges.convertTo(model_edge_magnitude, CV_8UC1, 255);
      ss.str("");
      ss << save_path << "_edge_magnitude_" << pyramid_level << "_" << rotation << ".png";
      cv::imwrite(ss.str(), model_edge_magnitude);
      ROS_INFO_STREAM("Saved magnitude image to "<<ss.str());

      // save edge derivatives (just for visualization)
      cv::Mat model_grad_x, model_grad_y;
      grad_x.setTo(0.0, hys_edges==0.0);
      grad_y.setTo(0.0, hys_edges==0.0);
      grad_x.convertTo(model_grad_x, CV_8UC1, 255);
      grad_y.convertTo(model_grad_y, CV_8UC1, 255);
      ss.str("");
      ss << save_path << "_grad_x_" << pyramid_level << "_" << rotation << ".png";
      cv::imwrite(ss.str(), model_grad_x);
      ROS_INFO_STREAM("Saved x gradients to "<<ss.str());
      ss.str("");
      ss << save_path << "_grad_y_" << pyramid_level << "_" << rotation << ".png";
      cv::imwrite(ss.str(), model_grad_y);
      ROS_INFO_STREAM("Saved y gradients to "<<ss.str());

      ss.str("");
      ss << "rotation_" << rotation;
      std::string replacement_string = ss.str();
      std::replace( replacement_string.begin(), replacement_string.end(), '.', '_');
      ROS_INFO_STREAM("ss str rot "<<replacement_string);
      fs << replacement_string << "{";
      fs << "coordinates_x" << template_coordinates_x;
      fs << "coordinates_y" << template_coordinates_y;
      fs << "magnitudes" << template_magnitudes;
      fs << "grad_x" << template_grad_x;
      fs << "grad_y" << template_grad_y;
      fs << "}";

      fs_mat << replacement_string << "{";
      fs_mat << "magnitude_mat" << model_edge_magnitude;
      fs_mat << "grad_x_mat" << model_grad_x;
      fs_mat << "grad_y_mat" << model_grad_y;
      fs_mat << "}";
    }
    fs << "}";
    fs_mat << "}";

    // move up one pyramid level
    cv::pyrDown(converted_reference_image, converted_reference_image);
    rotation_step *= 2.0;
  }
  fs << "}";
  fs_mat << "}";
  fs.release();
  fs_mat.release();
  return 0;
}
