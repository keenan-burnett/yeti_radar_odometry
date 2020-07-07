#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"

void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<float> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data) {
    int encoder_size = 5600;
    std::cout << path << std::endl;
    cv::Mat raw_example_data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int N = raw_example_data.rows;
    timestamps = std::vector<int64_t>(N, 0);
    azimuths = std::vector<float>(N, 0);
    valid = std::vector<bool>(N, true);
    int range_bins = 3768;
    fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
    std::cout << N << " " << raw_example_data.cols << std::endl;
    for (int i = 0; i < N; ++i) {
        uchar* byteArray = raw_example_data.ptr<uchar>(i);
        timestamps[i] = *((int64_t *)(byteArray));
        azimuths[i] = *((uint16_t *)(byteArray + 8)) * 2 * M_PI / float(encoder_size);
        valid[i] = byteArray[10] == 255;
        for (int j = 0; j < range_bins; j++) {
            fft_data.at<float>(i, j) = (float)*(byteArray + 11 + j) / 255.0;
        }
    }
}

void radar_polar_to_cartesian(std::vector<float> azimuths, cv::Mat fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img) {

    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    for (int j = 0; j < map_y.cols; ++j) {
        float m = -1 * cart_min_range + j * cart_resolution;
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = m;
        }
    }
    for (int i = 0; i < map_x.rows; ++i) {
        float m = cart_min_range - i * cart_resolution;
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = m;
        }
    }
    cv::Mat range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    float azimuth_step = azimuths[1] - azimuths[0];
    for (int i = 0; i < range.rows; ++i) {
        for (int j = 0; j < range.cols; ++j) {
            float x = map_x.at<float>(i, j);
            float y = map_y.at<float>(i, j);
            float r = (sqrt(pow(x, 2) + pow(y, 2)) - radar_resolution / 2) / radar_resolution;
            if (r < 0)
                r = 0;
            range.at<float>(i, j) = r;
            float theta = atan2f(y, x);
            if (theta < 0)
                theta += 2 * M_PI;
            angle.at<float>(i, j) = (theta - azimuths[0]) / azimuth_step;
        }
    }
    if (interpolate_crossover) {
        cv::Mat a0 = cv::Mat::zeros(1, fft_data.cols, CV_32F);
        cv::Mat aN_1 = cv::Mat::zeros(1, fft_data.cols, CV_32F);
        for (int j = 0; j < fft_data.cols; j++) {
            a0.at<float>(0, j) = fft_data.at<float>(0, j);
            aN_1.at<float>(0, j) = fft_data.at<float>(fft_data.rows-1, j);
        }
        cv::vconcat(aN_1, fft_data, fft_data);
        cv::vconcat(fft_data, a0, fft_data);
        angle = angle + 1;
    }
    cv::remap(fft_data, cart_img, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}
