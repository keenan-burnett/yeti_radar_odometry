#pragma once
#include <vector>
#include <opencv2/core.hpp>

/*!
   \brief Extract CFAR features from the polar radar data, one azimuth at a time (1D)
   \param fft_data Polar radar power readings
   \param window_size Length of window used to estimate local clutter power
   \param scale local clutter power is multiplied by scale
   \param guard_cells How many cells on the left and right side of the test cell are ignored before estimation
   \param min_range We ignore the range bins less than this
   \param [out] vector of feature locations (azimuth_bin, range_bin)
*/
void cfar1d(cv::Mat fft_data, int window_size, float scale, int guard_cells, int min_range,
    std::vector<cv::Point2f> & targets);

/*!
   \brief "Description"
   \param "Param description"
*/
void cen2018features(cv::Mat fft_data, float zq, int w_median, int sigma_gauss, int min_range,
    std::vector<cv::Point2f> &targets);
