#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>

/*!
   \brief Retrieves a vector of the (radar) file names in ascending order of time stamp
   \param datadir (absolute) path to the directory that contains (radar) files
   \param radar_files [out] A vector to be filled with a string for each file name
*/
void get_file_names(std::string datadir, std::vector<std::string> &radar_files);

/*!
   \brief Decode a single Oxford Radar RobotCar Dataset radar example
   \param path path to the radar image png file
   \param timestamps [out] Timestamp for each azimuth in int64 (UNIX time)
   \param azimuths [out] Rotation for each polar radar azimuth (radians)
   \param valid [out] Mask of whether azimuth data is an original sensor reasing or interpolated from adjacent azimuths
   \param fft_data [out] Radar power readings along each azimuth
*/
void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<float> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data);

/*!
   \brief Decode a single Oxford Radar RobotCar Dataset radar example
   \param azimuths Rotation for each polar radar azimuth (radians)
   \param fft_data Radar power readings along each azimuth
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width Width and height of the returned square cartesian output (pixels).
   \param interpolate_crossover If true, interpolates between the end and start azimuth of the scan.
   \param cart_img [out] Cartesian radar power readings
*/
void radar_polar_to_cartesian(std::vector<float> &azimuths, cv::Mat &fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img);

/*!
   \brief Converts points from polar coordinates to cartesian coordinates
   \param azimuths The actual azimuth of each row in the fft data reported by the Navtech sensor
   \param polar_points Matrix of point locations (azimuth_bin, range_bin) x N
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_points [out] Matrix of points in cartesian space (x, y) x N in metric
*/
void polar_to_cartesian_points(std::vector<float> azimuths, Eigen::MatrixXf polar_points, float radar_resolution,
    Eigen::MatrixXf &cart_points);

/*!
   \brief Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
   \param cart_points Vector of points in metric cartesian space (x, y)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width: Width and height of the returned square cartesian output (pixels)
   \param bev_points [out] Vector of pixel locations in the BEV cartesian image (u, v)
*/
void convert_to_bev(Eigen::MatrixXf cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point2f> &bev_points);

void convert_bev_to_polar(Eigen::MatrixXf bev_points, float cart_resolution, int cart_pixel_width,
    Eigen::MatrixXf &cart_points);

/*!
   \brief Draws a red dot for each feature on the top-down cartesian view of the radar image
   \param cart_img Cartesian radar power readings
   \param cart_targets Matrix of points in cartesian space (x, y) < N in metric
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width Width and height of the square cartesian image.
   \param vis [out] Output image with the features drawn onto it
*/
void draw_points(cv::Mat cart_img, Eigen::MatrixXf cart_targets, float cart_resolution, int cart_pixel_width,
    cv::Mat &vis);

/*!
   \brief Retrieves the ground truth odometry between radar timestamps t1 and t2
   \param gtfile (absolute) file location of the radar_odometry.csv file
   \param t1
   \param t2
   \param gt [out] Vector of floats for the ground truth transform between radar timestamp t1 and t2 (x, y, z, r, p, y)
*/
void get_groundtruth_odometry(std::string gtfile, int64 t1, int64 t2, std::vector<float> &gt);
