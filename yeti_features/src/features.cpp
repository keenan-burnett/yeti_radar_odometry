#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <features.hpp>

void cfar1d(cv::Mat fft_data, int window_size, float scale, int guard_cells, int min_range,
    std::vector<cv::Point2f> & targets) {
    assert(fft_data.depth() == CV_32F);
    assert(fft_data.channels() == 1);
    int kernel_size = window_size + guard_cells * 2 + 1;
    cv::Mat kernel = cv::Mat::ones(1, kernel_size, CV_32F) * -1 * scale / window_size;
    kernel.at<float>(0, kernel_size / 2) = 1;
    for (int i = 0; i < guard_cells; i++) {
        kernel.at<float>(0, window_size / 2 + i) = 0;
    }
    for (int i = 0; i < guard_cells; i++) {
        kernel.at<float>(0, kernel_size / 2 + 1 + i) = 0;
    }
    cv::Mat output;
    cv::filter2D(fft_data, output, -1, kernel, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
    // Find filter responses > 0
    targets.clear();
    for (int i = 0; i < output.rows; ++i) {
        for (int j = min_range; j < output.cols; j++) {
            if (output.at<float>(i, j) > 0) {
                targets.push_back(cv::Point(i, j));
            }
        }
    }
}

void cen2018features(cv::Mat fft_data, float zq, int sigma_gauss, int min_range, std::vector<cv::Point2f> &targets) {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> sigma_q(fft_data.rows, 0);
    // Estimate the bias and subtract it from the signal (0.015)
    cv::Mat q = fft_data.clone();
    int downsample = 4;
    std::vector<float> row(fft_data.cols / downsample);

    for (int i = 0; i < fft_data.rows; ++i) {
        for (uint j = 0; j < row.size(); ++j) {
            row[j] = fft_data.at<float>(i, j * downsample);
        }
        std::sort(row.begin(), row.end());
        float median = row[row.size() / 2];
        for (int j = 0; j < fft_data.cols; ++j) {
            q.at<float>(i, j) = fft_data.at<float>(i, j) - median;
        }
    }

    // Create 1D Gaussian Filter (0.09)
    assert(sigma_gauss % 2 == 1);
    int fsize = sigma_gauss * 3;
    int mu = fsize / 2;
    float sig_sqr = sigma_gauss * sigma_gauss;
    cv::Mat filter = cv::Mat::zeros(1, fsize, CV_32F);
    float s = 0;
    for (int i = 0; i < fsize; ++i) {
        filter.at<float>(0, i) = exp(-0.5 * (i - mu) * (i - mu) / sig_sqr);
        s += filter.at<float>(0, i);
    }
    filter /= s;
    cv::Mat p;
    cv::filter2D(q, p, -1, filter, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);

    // Estimate variance of noise at each azimuth (0.004)
    for (int i = 0; i < fft_data.rows; ++i) {
        int nonzero = 0;
        for (int j = 0; j < fft_data.cols; ++j) {
            float n = q.at<float>(i, j);
            if (n < 0) {
                sigma_q[i] += 2 * (n * n);
                nonzero++;
            }
        }
        if (nonzero)
            sigma_q[i] = sqrt(sigma_q[i] / nonzero);
        else
            sigma_q[i] = 0.034;
    }

    // Extract peak centers from each azimuth
    targets.clear();
#pragma omp declare reduction(merge : std::vector<cv::Point2f> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))  // NOLINT
#pragma omp parallel for reduction(merge: targets)
    for (int i = 0; i < fft_data.rows; ++i) {
        std::vector<int> peak_points;
        float thres = zq * sigma_q[i];
        for (int j = min_range; j < fft_data.cols; ++j) {
            float nqp = exp(-0.5 * pow((q.at<float>(i, j) - p.at<float>(i, j)) / sigma_q[i], 2));
            float npp = exp(-0.5 * pow(p.at<float>(i, j) / sigma_q[i], 2));
            float b = nqp - npp;
            float y = q.at<float>(i, j) * (1 - nqp) + p.at<float>(i, j) * b;
            if (y > thres) {
                peak_points.push_back(j);
            } else if (peak_points.size() > 0) {
                targets.push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
                peak_points.clear();
            }
        }
        if (peak_points.size() > 0)
            targets.push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    std::cout << "feature extraction: " << e.count() << std::endl;
}
