#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "radar_utils.hpp"

static inline bool exists(const std::string& name) {
    struct stat buffer;
    return !(stat (name.c_str(), &buffer) == 0);
}

struct less_than_img {
    inline bool operator() (const std::string& img1, const std::string& img2) {
        std::vector<std::string> parts;
        boost::split(parts, img1, boost::is_any_of("."));
        int64 i1 = std::stoll(parts[0]);
        boost::split(parts, img2, boost::is_any_of("."));
        int64 i2 = std::stoll(parts[0]);
        return i1 < i2;
    }
};

void get_file_names(std::string datadir, std::vector<std::string> &radar_files) {
    DIR *dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name))
            radar_files.push_back(dp->d_name);
    }
    // Sort files in ascending order of time stamp
    std::sort(radar_files.begin(), radar_files.end(), less_than_img());
}

void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<float> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data) {
    int encoder_size = 5600;
    cv::Mat raw_example_data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int N = raw_example_data.rows;
    timestamps = std::vector<int64_t>(N, 0);
    azimuths = std::vector<float>(N, 0);
    valid = std::vector<bool>(N, true);
    int range_bins = 3768;
    fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
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
        for (int j = 0; j < fft_data.cols; ++j) {
            a0.at<float>(0, j) = fft_data.at<float>(0, j);
            aN_1.at<float>(0, j) = fft_data.at<float>(fft_data.rows-1, j);
        }
        cv::vconcat(aN_1, fft_data, fft_data);
        cv::vconcat(fft_data, a0, fft_data);
        angle = angle + 1;
    }
    cv::remap(fft_data, cart_img, range, angle, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
}

void polar_to_cartesian_points(std::vector<float> azimuths, Eigen::MatrixXf polar_points,
    float radar_resolution, Eigen::MatrixXf &cart_points) {
    cart_points = polar_points;
    for (uint i = 0; i < polar_points.cols(); ++i) {
        float azimuth = azimuths[polar_points(0, i)];
        float r = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth);
        cart_points(1, i) = r * sin(azimuth);
    }
}

void convert_to_bev(Eigen::MatrixXf cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point> &bev_points) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    for (uint i = 0; i < cart_points.cols(); ++i) {
        int u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        int v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u && u < cart_pixel_width && 0 < v && v < cart_pixel_width)
            bev_points.push_back(cv::Point(u, v));
    }
}

void draw_points(cv::Mat cart_img, Eigen::MatrixXf cart_targets, float cart_resolution, int cart_pixel_width,
    cv::Mat &vis) {
    std::vector<cv::Point> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);
    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    for (cv::Point p : bev_points) {
        cv::circle(vis, p, 1, cv::Scalar(0, 0, 255), -1);
    }
}

void get_groundtruth_odometry(std::string gtfile, int64 t1, int64 t2, std::vector<float> &gt) {
    std::ifstream ifs(gtfile);
    std::string line;
    std::getline(ifs, line);
    gt.clear();
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (std::stoll(parts[9]) == t1 && std::stoll(parts[8]) == t2) {
            for (int i = 2; i < 8; ++i) {
                gt.push_back(std::stof(parts[i]));
            }
            break;
        }
    }
}
