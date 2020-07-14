#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <boost/algorithm/string.hpp>

struct less_than_img {
    inline bool operator() (const std::string& img1, const std::string& img2) {
        std::vector<std::string> parts;
        boost::split(parts, img1, boost::is_any_of("."));
        int i1 = std::stol(parts[0]);
        boost::split(parts, img2, boost::is_any_of("."));
        int i2 = std::stol(parts[0]);
        return i1 < i2;
    }
};

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
   \param cart_pixel_size Width and height of the returned square cartesian output (pixels).
   \param interpolate_crossover If true, interpolates between the end and start azimuth of the scan.
   \param cart_img [out] Cartesian radar power readings
*/
void radar_polar_to_cartesian(std::vector<float> azimuths, cv::Mat fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img);

/*!
   \brief Converts points from polar coordinates to cartesian coordinates
   \param azimuths The actual azimuth of each row in the fft data reported by the Navtech sensor
   \param polar_points Vector of point locations (azimuth_bin, range_bin)
   \param radar_resolution Resolution of the polar radar data (metres per pixel)
   \param cart_points [out] Vector of points in cartesian space (x, y) in metric
*/
void polar_to_cartesian_points(std::vector<float> azimuths, std ::vector<cv::Point2f> polar_points,
    float radar_resolution, std::vector<cv::Point2f> &cart_points);

/*!
   \brief Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
   \param cart_points Vector of points in metric cartesian space (x, y)
   \param cart_resolution Cartesian resolution (meters per pixel)
   \param cart_pixel_width: Width and height of the returned square cartesian output (pixels)
   \param bev_points [out] Vector of pixel locations in the BEV cartesian image (u, v)
*/
void convert_to_bev(std::vector<cv::Point2f> cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point> &bev_points);
