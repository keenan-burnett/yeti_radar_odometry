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
void cfar1d(cv::Mat fft_data, int window_size, float scale, int guard_cells, std::vector<cv::Point> & targets) {
    assert(fft_data.depth == CV_32F);
    assert(fft_data.channels() == 1);
    int kernel_size = window_size + guard_cells * 2 + 1;
    cv::Mat kernel = cv::Mat::ones(1, kernel_size, CV_32F) * -1 * scale / window_size;
    kernel(0, kernel_size / 2) = 1;
    for (int i = 0; i < guard_cells; i++) {
        kernel(0, window_size / 2 + i) = 0;
    }
    for (int i = 0; i < guard_cells; i++) {
        kernel(0, kernel_size / 2 + 1 + i) = 0;
    }
    int ddepth = -1;    // Output has same channels as input
    cv::Point anchor = cv::Point(-1, -1);   // Anchor is kernel center
    int delta = 0;      // Optional value added to filtered pixels
    cv::Mat output;
    cv::filter2D(fft_data, output, -1, kernel, cv::Point(-1, -1), cv::BORDER_REFLECT101);
    // int nRows = fft_data.rows;
    // int nCols = fft_data.cols;
    // if (fft_data.isContinuous()) {
    //     nCols *= nRows;
    //     nRows = 1;
    // }
    // Find filter responses > 0
    targets.clear();
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; j++) {
            if (output(i, j) > 0) {
                targets.push_back(cv::Point(i, j));
            }
        }
    }
}
