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

static void medianFilter1d(cv::Mat src, cv::Mat &dst, int window_size) {
    dst = cv::Mat::zeros(src.rows, src.cols, CV_32F);
    int w2 = window_size / 2;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            std::vector<float> window;
            for (int k = j-w2; k <= j+w2; ++k) {
                if (k < 0)
                    window.push_back(src.at<float>(i, 0));
                else if (k > src.cols - 1)
                    window.push_back(src.at<float>(i, src.cols - 1));
                else
                    window.push_back(src.at<float>(i, k));
            }
            std::sort(window.begin(), window.end());
            dst.at<float>(i, j) = window[w2];
        }
    }
}

static void norm(cv::Mat x, std::vector<float> sigma, cv::Mat &y) {
    y = cv::Mat::zeros(x.rows, x.cols, CV_32F);
    for (int i = 0; i < x.rows; ++i) {
        float denom = sigma[i] * sqrt(2 * M_PI);
        for (int j = 0; j < x.cols; ++j) {
            y.at<float>(i, j) = exp(-0.5 * pow(x.at<float>(i, j) / sigma[i], 2)) / denom;
        }
    }
}

void cen2018features(cv::Mat fft_data, float zq, int w_median, int sigma_gauss, int min_range,
    std::vector<cv::Point2f> &targets) {

    // Remove bias from each azimuth by calculating median power level (median filter too slow)
    cv::Mat q = cv::Mat::zeros(fft_data.rows, fft_data.cols, CV_32F);
    std::vector<float> row(fft_data.cols, 0);

    float z_upper = 2.0;

    std::vector<float> sigma_q(fft_data.rows, 0);

    // auto t1 = std::chrono::high_resolution_clock::now();

    // for (int i = 0; i < fft_data.rows; ++i) {
    //     // Calculate mean
    //     float mean = 0;
    //     for (int j = 0; j < fft_data.cols; ++j) {
    //         mean += fft_data.at<float>(i, j);
    //     }
    //     mean /= fft_data.cols;
    //     float variance = 0;
    //     for (int j = 0; j < fft_data.cols; ++j) {
    //         variance += pow(fft_data.at<float>(i, j) - mean, 2);
    //     }
    //     variance /= fft_data.cols;
    //     float sigma = sqrt(variance);
    //     // Recalculate mean and variance using only bottom 95% of points
    //     float mean2 = 0;
    //     int count = 0;
    //     float upper = z_upper * sigma;
    //     for (int j = 0; j < fft_data.cols; ++j) {
    //         if (fft_data.at<float>(i, j) < upper) {
    //             mean2 += fft_data.at<float>(i, j);
    //             count++;
    //         }
    //     }
    //     mean2 /= count;
    //     float variance2 = 0;
    //     for (int j = 0; j < fft_data.cols; ++j) {
    //         if (fft_data.at<float>(i, j) < upper) {
    //             variance2 += pow(fft_data.at<float>(i, j) - mean2, 2);
    //         }
    //         q.at<float>(i, j) = fft_data.at<float>(i, j) - mean2;
    //     }
    //     variance2 /= count;
    //     sigma_q[i] = sqrt(variance2);
    // }

    for (int i = 0; i < fft_data.rows; ++i) {
        for (int j = 0; j < fft_data.cols; ++j) {
            row[j] = fft_data.at<float>(i, j);
        }
        std::sort(row.begin(), row.end());
        float median = row[row.size() / 2];
        for (int j = 0; j < fft_data.cols; ++j) {
            q.at<float>(i, j) = fft_data.at<float>(i, j) - median;
        }
    }

    // auto t2 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> e = t2 - t1;
    // std::cout << e.count() << std::endl;

    // Create 1D Gaussian Filter
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

    // Estimate variance of noise at each azimuth
    // std::vector<float> sigma_q(fft_data.rows, 0);
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

    cv::Mat nqp, npp, nzero;
    norm(p, sigma_q, npp);
    norm(q - p, sigma_q, nqp);
    norm(cv::Mat::zeros(fft_data.rows, 1, CV_32F), sigma_q, nzero);
    cv::Mat ones = cv::Mat::ones(fft_data.rows, fft_data.cols, CV_32F);

    cv::Mat b = nqp - npp;

    for (int i = 0; i < b.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            nqp.at<float>(i, j) /= nzero.at<float>(i, 0);
            b.at<float>(i, j) /= nzero.at<float>(i, 0);
        }
    }

    cv::Mat y = q.mul(ones - nqp) + p.mul(b);

    // Threshold
    targets.clear();
    for (int i = 0; i < fft_data.rows; ++i) {
        std::vector<int> peak_points;
        for (int j = 0; j < fft_data.cols; ++j) {
            if (y.at<float>(i, j) > zq * sigma_q[i]) {
                peak_points.push_back(j);
            } else if (peak_points.size() > 0) {
                targets.push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
                peak_points.clear();
            }
        }
        if (peak_points.size() > 0)
            targets.push_back(cv::Point(i, peak_points[peak_points.size() / 2]));
    }
}
