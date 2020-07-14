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

// Runtime: 0.038s
void cen2018features(cv::Mat fft_data, float zq, int sigma_gauss, int min_range, std::vector<cv::Point2f> &targets) {
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> sigma_q(fft_data.rows, 0);
    // Estimate the bias and subtract it from the signal
    cv::Mat q = fft_data.clone();
    for (int i = 0; i < fft_data.rows; ++i) {
        float mean = 0;
        for (int j = 0; j < fft_data.cols; ++j) {
            mean += fft_data.at<float>(i, j);
        }
        mean /= fft_data.cols;
        for (int j = 0; j < fft_data.cols; ++j) {
            q.at<float>(i, j) = fft_data.at<float>(i, j) - mean;
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

struct Point {
    float i;
    int a;
    int r;
    Point(float i_, int a_, int r_) {i = i_; a = a_; r = r_;}
};

struct greater_than_pt {
    inline bool operator() (const Point& p1, const Point& p2) {
        return p1.i > p2.i;
    }
};

static void findRangeBoundaries(cv::Mat &s, int a, int r, int &rlow, int &rhigh) {
    rlow = r;
    rhigh = r;
    if (r > 0) {
        for (int i = r - 1; i >= 0; i--) {
            if (s.at<float>(a, i) < 0)
                rlow = i;
            else
                break;
        }
    }
    if (r < s.rows - 1) {
        for (int i = r + 1; i < s.cols; i++) {
            if (s.at<float>(a, i) < 0)
                rhigh = i;
            else
                break;
        }
    }
}

static bool checkAdjacentMarked(cv::Mat &R, int a, int start, int end) {
    int below = a - 1;
    int above = a + 1;
    if (below < 0)
        below = R.rows - 1;
    if (above >= R.rows)
        above = 0;
    for (int r = start; r <= end; r++) {
        if (R.at<float>(below, r) || R.at<float>(above, r))
            return true;
    }
    return false;
}

static void getMaxInRegion(cv::Mat &h, int a, int start, int end, int &max_r) {
    int max = -1000;
    for (int r = start; r <= end; r++) {
        if (h.at<float>(a, r) > max) {
            max = h.at<float>(a, r);
            max_r = r;
        }
    }
}

// Runtime: 0.050s
void cen2019features(cv::Mat fft_data, int max_points, int min_range, std::vector<cv::Point2f> &targets) {
    auto t1 = std::chrono::high_resolution_clock::now();
    // Calculate gradient along each azimuth using the Prewitt operator
    cv::Mat prewitt = cv::Mat::zeros(1, 3, CV_32F);
    prewitt.at<float>(0, 0) = -1;
    prewitt.at<float>(0, 2) = 1;
    cv::Mat g;
    cv::filter2D(fft_data, g, -1, prewitt, cv::Point(-1, -1), 0, cv::BORDER_REFLECT101);
    g = cv::abs(g);
    double maxg = 1, ming = 1;
    cv::minMaxIdx(g, &ming, &maxg);
    g /= maxg;

    // Subtract the mean from the radar data and scale it by 1 - gradient magnitude
    float mean = cv::mean(fft_data)[0];
    cv::Mat s = fft_data - mean;
    cv::Mat h = s.mul(1 - g);
    float mean_h = cv::mean(h)[0];

    // Get indices in descending order of intensity
    std::vector<Point> vec;
    for (int i = 0; i < fft_data.rows; ++i) {
        for (int j = 0; j < fft_data.cols; ++j) {
            if (h.at<float>(i, j) > mean_h)
                vec.push_back(Point(h.at<float>(i, j), i, j));
        }
    }
    std::sort(vec.begin(), vec.end(), greater_than_pt());

    // Create a matrix, R, of "marked" regions consisting of continuous regions of an azimuth that may contain a target
    int false_count = fft_data.rows * fft_data.cols;
    uint j = 0;
    int l = 0;
    cv::Mat R = cv::Mat::zeros(fft_data.rows, fft_data.cols, CV_32F);
    while (l < max_points && j < vec.size() && false_count > 0) {
        if (!R.at<float>(vec[j].a, vec[j].r)) {
            int rlow = vec[j].r;
            int rhigh = vec[j].r;
            findRangeBoundaries(s, vec[j].a, vec[j].r, rlow, rhigh);
            bool already_marked = false;
            for (int i = rlow; i <= rhigh; i++) {
                if (R.at<float>(vec[j].a, i)) {
                    already_marked = true;
                    continue;
                }
                R.at<float>(vec[j].a, i) = 1;
                false_count--;
            }
            if (!already_marked)
                l++;
        }
        j++;
    }

    targets.clear();
#pragma omp declare reduction(merge : std::vector<cv::Point2f> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))  // NOLINT
#pragma omp parallel for reduction(merge: targets)
    for (int i = 0; i < fft_data.rows; i++) {
        // Find the continuous marked regions in each azimuth
        int start = 0;
        int end = 0;
        bool counting = false;
        for (int j = min_range; j < fft_data.cols; j++) {
            if (R.at<float>(i, j)) {
                if (!counting) {
                    start = j;
                    end = j;
                    counting = true;
                } else {
                    end = j;
                }
            } else if (counting) {
                // Check whether adjacent azimuths contain a marked pixel in this range region
                if (checkAdjacentMarked(R, i, start, end)) {
                    int max_r = start;
                    getMaxInRegion(h, i, start, end, max_r);
                    targets.push_back(cv::Point(i, max_r));
                }
                counting = false;
            }
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> e = t2 - t1;
    std::cout << "feature extraction: " << e.count() << std::endl;
}
