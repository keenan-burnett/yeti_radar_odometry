#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include "matplotlibcpp.h"  // NOLINT
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "association.hpp"
#include <nanoflann.hpp>
namespace plt = matplotlibcpp;
using namespace nanoflann;  // NOLINT

template <typename T>
struct PointCloud {
    struct Point {
        T  x, y, z;
    };
    std::vector<Point>  pts;
    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
    if (dim == 0) return pts[idx].x;
    else if (dim == 1) return pts[idx].y;
    else
        return pts[idx].z;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

double wrap2pi(double theta) {
    if (theta < 0) {
        return theta + 2 * M_PI;
    } else if (theta > 2 * M_PI) {
        return theta - 2 * M_PI;
    } else {
        return theta;
    }
}

template <typename T>
bool contains(std::vector<T> v, T x) {
    for (uint i = 0; i < v.size(); ++i) {
        if (v[i] == x)
            return true;
    }
    return false;
}

void get_shape(std::vector<double> vertices_x, std::vector<double> vertices_y, std::vector<double> &shape_x,
    std::vector<double> &shape_y, double resolution) {
    assert(vertices_x.size() == vertices_y.size());
    for (uint i = 0; i < vertices_x.size() - 1; ++i) {
        double x1 = vertices_x[i];
        double y1 = vertices_y[i];
        double x2 = vertices_x[i + 1];
        double y2 = vertices_y[i + 1];
        double delta_x = x2 - x1;
        double delta_y = y2 - y1;
        double d = sqrt(pow(delta_x, 2) + pow(delta_y, 2));
        int n = d / resolution;
        shape_x.push_back(x1);
        shape_y.push_back(y1);
        for (int j = 0; j < n; j++) {
            shape_x.push_back(x1 + j * delta_x / double(n));
            shape_y.push_back(y1 + j * delta_y / double(n));
        }
    }
}

typedef KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> my_kd_tree_t;

int main(int argc, char *argv[]) {
    // Generate fake data using a known motion distortion
    double v = 20.0;
    double omega = 90.0 * M_PI / 180;  // rad/s
    if (argc > 1)
        v = atof(argv[1]);
    if (argc > 2)
        omega = atof(argv[2]);
    std::cout << v << " " << omega << std::endl;
    double resolution = 0.1;
    std::vector<double> square_x = {25, -25, -25, 25, 25};
    std::vector<double> square_y = {25, 25, -25, -25, 25};
    std::vector<double> cross_x = {25, 25, -25, -25, -75, -75, -25, -25, 25, 25, 75, 75, 25};
    std::vector<double> cross_y = {25, 75,  75,  25, 25, -25, -25, -75, -75, -25, -25, 25, 25};
    plt::plot(square_x, square_y, "k");

    std::vector<double> shape_x, shape_y;
    get_shape(square_x, square_y, shape_x, shape_y, resolution);
    PointCloud<double> cloud;
    cloud.pts.resize(shape_x.size());
    for (uint i = 0; i < shape_x.size(); ++i) {
        cloud.pts[i].x = shape_x[i];
        cloud.pts[i].y = shape_y[i];
        cloud.pts[i].z = 0;
    }

    my_kd_tree_t   index(3, cloud, KDTreeSingleIndexAdaptorParams(10));
    index.buildIndex();

    std::vector<double> x1, y1, desc1, x2, y2, desc2;
    std::vector<int64_t> t1, t2;
    std::vector<float> a1, a2;
    double delta_t = 0.000625;
    double time = 0.0;
    double search_increment = 0.25;
    double search_distance = 100.0;
    const double search_radius = 0.5;

    std::vector<double> x_pos_vec, y_pos_vec;

    // Simulate the generation of two clouds, motion-distorted:
    for (int scan = 0; scan < 2; scan++) {
        for (int i = 0; i < 400; ++i) {
            time += delta_t;
            // Get sensor position
            double theta_pos = omega * time;
            theta_pos = wrap2pi(theta_pos);
            double x_pos = 0, y_pos = 0;
            if (omega == 0) {
                x_pos = v * time;
                y_pos = 0;
            } else {
                x_pos = (v / omega) * sin(theta_pos);
                y_pos = (v / omega) * (1 - cos(theta_pos));
            }
            x_pos_vec.push_back(x_pos);
            y_pos_vec.push_back(y_pos);
            std::cout << "theta: " << theta_pos << " x_pos: " << x_pos << " ypos: " << y_pos << std::endl;

            double theta_rad = i * 0.9 * M_PI / 180.0;
            double theta = theta_pos + theta_rad;
            theta = wrap2pi(theta);
            double m = 0;
            double b = 0;
            bool flag = 0;

            if (scan == 0) {
                a1.push_back(theta_rad);
                t1.push_back(time * 1000000);
            } else {
                a2.push_back(theta_rad);
                t2.push_back(time * 1000000);
            }

            if ((0 <= theta && theta < M_PI / 4) || (3 * M_PI / 4 <= theta && theta < 5 * M_PI / 4) ||
                (7 * M_PI / 4 <= theta && theta < 2 * M_PI)) {
                m = tan(theta);  // y = m*x + b
                b = y_pos - m * x_pos;
            } else {
                m = cos(theta) / sin(theta);  // x = m*y + b
                b = x_pos - m * y_pos;
                flag = 1;
            }
            // Line search to find closest point
            double x_search = x_pos, y_search = y_pos;
            int n = search_distance / search_increment;
            for (int j = 0; j < n; ++j) {
                double query_pt[3] = {x_search, y_search, 0};
                std::vector<std::pair<size_t, double>> ret_matches;
                nanoflann::SearchParams params;
                const size_t nMatches = index.radiusSearch(&query_pt[0], search_radius, ret_matches, params);
                if (nMatches > 0) {
                    int idx = ret_matches[0].first;
                    double r = sqrt(pow(x_pos - cloud.pts[idx].x, 2) + pow(y_pos - cloud.pts[idx].y, 2));
                    std::cout << "r: " << r << std::endl;
                    if (scan == 0) {
                        desc1.push_back(double(idx));
                        x1.push_back(r * cos(theta_rad));
                        y1.push_back(r * sin(theta_rad));
                    } else {
                        desc2.push_back(double(idx));
                        x2.push_back(r * cos(theta_rad));
                        y2.push_back(r * sin(theta_rad));
                    }
                    break;
                }
                if (flag == 0) {
                    if (M_PI / 2 <= theta && theta < 3 * M_PI / 2)
                        x_search -= search_increment;
                    else
                        x_search += search_increment;
                    y_search =  m * x_search + b;
                } else if (flag == 1) {
                    if (0 <= theta && theta < M_PI)
                        y_search += search_increment;// using namespace Nabo;  // NOLINT
                    else
                        y_search -= search_increment;
                    x_search = m * y_search + b;
                }
            }
        }
    }

    std::map<std::string, std::string> kw;
    kw.insert(std::pair<std::string, std::string>("c", "r"));
    plt::scatter(x1, y1, 25.0, kw);
    plt::scatter(x_pos_vec, y_pos_vec, 25.0);
    std::map<std::string, std::string> kw2;
    kw2.insert(std::pair<std::string, std::string>("c", "b"));
    plt::scatter(x2, y2, 25.0, kw2);

    // Perform NN matching using the descriptors from each cloud:
    PointCloud<double> cloud2;
    cloud2.pts.resize(desc2.size());
    for (uint i = 0; i < desc2.size(); ++i) {
        cloud2.pts[i].x = desc2[i];
        cloud2.pts[i].y = 0;
        cloud2.pts[i].z = 0;
    }
    my_kd_tree_t index2(1, cloud2, KDTreeSingleIndexAdaptorParams(10));
    index2.buildIndex();
    std::vector<int> matches;
    size_t num_results = 1;
    std::vector<size_t>   ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);

    int size = 0;
    for (uint i = 0; i < desc1.size(); ++i) {
        double query_pt[3] = {desc1[i], 0, 0};
        num_results = index2.knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        if (!contains(matches, int(ret_index[0]))) {
            matches.push_back(ret_index[0]);
            size++;
        } else {
            matches.push_back(-1);
        }
        // std::cout << "d1: " << desc1[i] << " d2: " << matches[i] << std::endl;
    }

    // Create the p1 and p2 matrices based on the matches:
    Eigen::MatrixXf p1, p2;
    p1 = Eigen::MatrixXf::Zero(2, size);
    p2 = p1;
    int j = 0;
    for (uint i = 0; i < desc1.size(); ++i) {
        if (matches[i] == -1)
            continue;
        p1(0, j) = x1[i];
        p1(1, j) = y1[i];
        p2(0, j) = x2[matches[i]];
        p2(1, j) = y2[matches[i]];
        j++;
    }
    // run the motion-distorted RANSAC to extract the desired transform, motion parameters?
    Ransac<float> ransac(p2, p1, 0.35, 0.90, 100);
    ransac.computeModel();
    Eigen::MatrixXf T;
    ransac.getTransform(T);
    std::cout << "T: " << std::endl << T << std::endl;
    Eigen::MatrixXf p2prime = Eigen::MatrixXf::Ones(3, p2.cols());
    p2prime.block(0, 0, 2, p2.cols()) = p2;
    p2prime = T * p2prime;
    std::vector<double> x3, y3;
    for (uint i = 0; i < p2.cols(); ++i) {
        x3.push_back(p2prime(0, i));
        y3.push_back(p2prime(1, i));
    }
    std::map<std::string, std::string> kw3;
    kw3.insert(std::pair<std::string, std::string>("c", "g"));
    plt::scatter(x3, y3, 25.0, kw3);
    plt::show();

    // run the motion-distorted RANSAC to extract the motion parameters:
    MotionDistortedRansac mdransac(p1, p2, a1, a2, t1, t2, 0.35, 0.90, 100);
    std::cout << 1 << std::endl;
    mdransac.computeModel();
    Eigen::VectorXf w;
    mdransac.getMotion(w);
    std::cout << "w: " << std::endl << w << std::endl;

    return 0;
}
