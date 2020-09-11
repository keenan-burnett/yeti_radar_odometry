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

void get_file_names(std::string datadir, std::vector<std::string> &radar_files, std::string extension) {
    DIR *dirp = opendir(datadir.c_str());
    struct dirent *dp;
    while ((dp = readdir(dirp)) != NULL) {
        if (exists(dp->d_name)) {
            if (!extension.empty()) {
                std::vector<std::string> parts;
                boost::split(parts, dp->d_name, boost::is_any_of("."));
                if (parts[parts.size() - 1].compare(extension) != 0)
                    continue;
            }
            radar_files.push_back(dp->d_name);
        }
    }
    // Sort files in ascending order of time stamp
    std::sort(radar_files.begin(), radar_files.end(), less_than_img());
}

void load_radar(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    std::vector<bool> &valid, cv::Mat &fft_data) {
    int encoder_size = 5600;
    cv::Mat raw_example_data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    int N = raw_example_data.rows;
    timestamps = std::vector<int64_t>(N, 0);
    azimuths = std::vector<double>(N, 0);
    valid = std::vector<bool>(N, true);
    int range_bins = 3768;
    fft_data = cv::Mat::zeros(N, range_bins, CV_32F);
    for (int i = 0; i < N; ++i) {
        uchar* byteArray = raw_example_data.ptr<uchar>(i);
        timestamps[i] = *((int64_t *)(byteArray));
        azimuths[i] = *((uint16_t *)(byteArray + 8)) * 2 * M_PI / double(encoder_size);
        valid[i] = byteArray[10] == 255;
        for (int j = 0; j < range_bins; j++) {
            fft_data.at<float>(i, j) = (float)*(byteArray + 11 + j) / 255.0;
        }
    }
}

void load_velodyne(std::string path, std::vector<int64_t> &timestamps, std::vector<double> &azimuths,
    Eigen::MatrixXd &pc) {
    double hdl32e_range_resolution = 0.002;
    cv::Mat data = cv::imread(path, cv::IMREAD_GRAYSCALE);
    uint N = data.cols;
    cv::transpose(data, data);
    timestamps = std::vector<int64_t>(N * 32, 0);
    azimuths = std::vector<double>(N * 32, 0.0);
    Eigen::MatrixXd ranges = Eigen::MatrixXd::Zero(N, 32);
    for (uint i = 0; i < N; ++i) {
        uchar* byteArray = data.ptr<uchar>(i);
        uint16_t azimuth = *( (uint16_t *)(byteArray + 96));
        double a = double(azimuth) * M_PI / 18000.0;
        int64_t t = *((int64_t *)(byteArray + 98));
        int k = 0;
        for (uint j = 32; j < 96; j += 2) {
            uint16_t range = *( (uint16_t *)(byteArray + j));
            ranges(i, k) = double(range) * hdl32e_range_resolution;
            azimuths[i * 32 + k] = a;
            timestamps[i * 32 + k] = t;
            k++;
        }
    }
    // Convert to 3D point cloud
    std::vector<double> elevations = {-0.1862, -0.1628, -0.1396, -0.1164, -0.0930, -0.0698, -0.0466, -0.0232, 0.,
        0.0232, 0.0466, 0.0698, 0.0930, 0.1164, 0.1396, 0.1628, 0.1862, 0.2094, 0.2327, 0.2560, 0.2793, 0.3025, 0.3259,
        0.3491, 0.3723, 0.3957, 0.4189, 0.4421, 0.4655, 0.4887, 0.5119, 0.5353};
    double hdl32e_base_to_fire_height = 0.090805;
    // double hdl32e_minimum_range = 1.0;
    std::vector<double> x, y, z;
    for (uint i = 0; i < N; ++i) {
        for (uint j = 0; j < 32; ++j) {
            z.push_back(sin(elevations[j]) * ranges(i, j) - hdl32e_base_to_fire_height);
            double xy = cos(elevations[j]) * ranges(i, j);
            x.push_back(sin(azimuths[i * 32 + j]) * xy);
            y.push_back(-cos(azimuths[i * 32 + j]) * xy);
        }
    }
    pc = Eigen::MatrixXd::Zero(3, z.size());
    for (uint i = 0; i < z.size(); ++i) {
        pc(0, i) = x[i];
        pc(1, i) = y[i];
        pc(2, i) = z[i];
    }
}

void radar_polar_to_cartesian(std::vector<double> &azimuths, cv::Mat &fft_data, float radar_resolution,
    float cart_resolution, int cart_pixel_width, bool interpolate_crossover, cv::Mat &cart_img, int output_type) {

    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;

    cv::Mat map_x = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat map_y = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

#pragma omp parallel for collapse(2)
    for (int j = 0; j < map_y.cols; ++j) {
        for (int i = 0; i < map_y.rows; ++i) {
            map_y.at<float>(i, j) = -1 * cart_min_range + j * cart_resolution;
        }
    }
#pragma omp parallel for collapse(2)
    for (int i = 0; i < map_x.rows; ++i) {
        for (int j = 0; j < map_x.cols; ++j) {
            map_x.at<float>(i, j) = cart_min_range - i * cart_resolution;
        }
    }
    cv::Mat range = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);
    cv::Mat angle = cv::Mat::zeros(cart_pixel_width, cart_pixel_width, CV_32F);

    double azimuth_step = azimuths[1] - azimuths[0];
#pragma omp parallel for collapse(2)
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
    if (output_type == CV_8UC1) {
        double min, max;
        cv::minMaxLoc(cart_img, &min, &max);
        cart_img.convertTo(cart_img, CV_8UC1, 255.0 / max);
    }
}

void polar_to_cartesian_points(std::vector<double> azimuths, Eigen::MatrixXd polar_points,
    float radar_resolution, Eigen::MatrixXd &cart_points) {
    cart_points = polar_points;
    for (uint i = 0; i < polar_points.cols(); ++i) {
        double azimuth = azimuths[polar_points(0, i)];
        double r = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth);
        cart_points(1, i) = r * sin(azimuth);
    }
}

void polar_to_cartesian_points(std::vector<double> azimuths, std::vector<int64_t> times, Eigen::MatrixXd polar_points,
    float radar_resolution, Eigen::MatrixXd &cart_points, std::vector<int64_t> &point_times) {
    cart_points = polar_points;
    point_times = std::vector<int64_t>(polar_points.cols());
    for (uint i = 0; i < polar_points.cols(); ++i) {
        double azimuth = azimuths[polar_points(0, i)];
        double r = polar_points(1, i) * radar_resolution + radar_resolution / 2;
        cart_points(0, i) = r * cos(azimuth);
        cart_points(1, i) = r * sin(azimuth);
        point_times[i] = times[polar_points(0, i)];
    }
}

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width,
    std::vector<cv::Point2f> &bev_points) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    int j = 0;
    for (uint i = 0; i < cart_points.cols(); ++i) {
        double u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u && u < cart_pixel_width && 0 < v && v < cart_pixel_width) {
            bev_points.push_back(cv::Point2f(u, v));
            cart_points(0, j) = cart_points(0, i);
            cart_points(1, j) = cart_points(1, i);
            j++;
        }
    }
    cart_points.conservativeResize(2, bev_points.size());
}

void convert_to_bev(Eigen::MatrixXd &cart_points, float cart_resolution, int cart_pixel_width, int patch_size,
    std::vector<cv::KeyPoint> &bev_points, std::vector<int64_t> &point_times) {
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    bev_points.clear();
    int j = 0;
    for (uint i = 0; i < cart_points.cols(); ++i) {
        double u = (cart_min_range + cart_points(1, i)) / cart_resolution;
        double v = (cart_min_range - cart_points(0, i)) / cart_resolution;
        if (0 < u - patch_size && u + patch_size < cart_pixel_width && 0 < v - patch_size &&
            v + patch_size < cart_pixel_width) {
            bev_points.push_back(cv::KeyPoint(u, v, patch_size));
            point_times[j] = point_times[i];
            cart_points(0, j) = cart_points(0, i);
            cart_points(1, j) = cart_points(1, i);
            j++;
        }
    }
    point_times.resize(bev_points.size());
    cart_points.conservativeResize(2, bev_points.size());
}

void convert_from_bev(std::vector<cv::KeyPoint> bev_points, float cart_resolution, int cart_pixel_width,
    Eigen::MatrixXd &cart_points) {
    cart_points = Eigen::MatrixXd::Zero(2, bev_points.size());
    float cart_min_range = (cart_pixel_width / 2) * cart_resolution;
    if (cart_pixel_width % 2 == 0)
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution;
    for (uint i = 0; i < bev_points.size(); ++i) {
        cart_points(0, i) = cart_min_range - cart_resolution * bev_points[i].pt.y;
        cart_points(1, i) = cart_resolution * bev_points[i].pt.x - cart_min_range;
    }
}

void draw_points(cv::Mat cart_img, Eigen::MatrixXd cart_targets, float cart_resolution, int cart_pixel_width,
    cv::Mat &vis) {
    std::vector<cv::Point2f> bev_points;
    convert_to_bev(cart_targets, cart_resolution, cart_pixel_width, bev_points);
    cv::cvtColor(cart_img, vis, cv::COLOR_GRAY2BGR);
    for (cv::Point2f p : bev_points) {
        // cv::circle(vis, p, 1, cv::Scalar(0, 0, 255), -1);
        vis.at<cv::Vec3f>(int(p.y), int(p.x)) = cv::Vec3f(0, 0, 255);
    }
}

bool get_groundtruth_odometry(std::string gtfile, int64 t1, int64 t2, std::vector<float> &gt) {
    std::ifstream ifs(gtfile);
    std::string line;
    std::getline(ifs, line);
    gt.clear();
    bool gtfound = false;
    while (std::getline(ifs, line)) {
        std::vector<std::string> parts;
        boost::split(parts, line, boost::is_any_of(","));
        if (std::stoll(parts[9]) == t1 && std::stoll(parts[8]) == t2) {
            for (int i = 2; i < 8; ++i) {
                gt.push_back(std::stof(parts[i]));
            }
            gtfound = true;
            break;
        }
    }
    return gtfound;
}
