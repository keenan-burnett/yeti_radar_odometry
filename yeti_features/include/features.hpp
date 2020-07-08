#pragma once
#include <Eigen/Geometry>
#include <vector>
#include <opencv2/core.hpp>

/*!
   \brief "Description"
   \param "Param description"
   \pre "Pre-conditions"
   \post "Post-conditions"
   \return "Return of the function"
*/
void cfar1d(cv::Mat fft_data, int window_size, float scale, int guard_cells, std::vector<cv::Point2f> & targets);
