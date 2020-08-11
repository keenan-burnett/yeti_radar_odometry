#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include "matplotlibcpp.h"  // NOLINT
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "association.hpp"
namespace plt = matplotlibcpp;

double wrap2pi(double theta) {
    if (theta < 0) {
        return theta + 2 * M_PI;
    } else if (theta > 2 * M_PI) {
        return theta - 2 * M_PI;
    } else {
        return theta;
    }
}

void getClosest(double m, double b, bool flag, double x_pos, double y_pos, double x_test, double y_test,
    double &x_closest, double &y_closest) {
    double x2 = 0, y1 = 0;
    if (!flag) {
        // y = m * x + b
        // test 1 x = x_test
        y1 = m * x_test + b;
        // test 2 y = y_test
        x2 = (y_test - b) / m;
    } else {
        // x = m * y + b
        // test 1 x = x_test
        y1 = (x_test - b) / m;
        // test 2 y = y_test
        x2 = m * y_test + b;
    }
    double d1 = pow(y_pos - y1, 2) + pow(x_pos - x_test, 2);
    double d2 = pow(x_pos - x2, 2) + pow(y_pos - y_test, 2);
    if (d1 < d2) {
        x_closest = x_test;
        y_closest = y1;
    } else {
        x_closest = x2;
        y_closest = y_test;
    }
}

int main(int argc, char *argv[]) {
    // Generate fake data using a known motion distortion
    float v = 20.0;
    float omega = 90.0 * M_PI / 180;  // rad/s
    if (argc > 1)
        v = atof(argv[1]);
    if (argc > 2)
        omega = atof(argv[2]);

    std::cout << v << " " << omega << std::endl;

    // int n = 2000;
    // std::vector<double> square_x(n), square_y(n);
    // for (int i = 0; i < 500; ++i) {
    //
    // }

    std::vector<double> square_x = {25, -25, -25, 25, 25};
    std::vector<double> square_y = {25, 25, -25, -25, 25};
    plt::plot(square_x, square_y, "k");



    std::vector<double> x, y;
    double delta_t = 0.000625;
    double time = 0.25;
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

        double theta = theta_pos + i * 0.9 * M_PI / 180.0;
        double m = 0;
        double b = 0;
        bool flag = 0;

        if ((0 <= theta && theta < M_PI / 4) || (3 * M_PI / 4 <= theta && theta < 5 * M_PI / 4) ||
            (7 * M_PI / 4 <= theta && theta < 2 * M_PI)) {
            m = tan(theta);  // y = m*x + b
            b = y_pos - m * x_pos;
        } else {
            m = cos(theta) / sin(theta);  // x = m*y + b
            b = x_pos - m * y_pos;
            flag = 1;
        }
        double x2 = x_pos, y2 = y_pos;
        if (0 <= theta && theta < M_PI / 2) {
            getClosest(m, b, flag, x_pos, y_pos, 25, 25, x2, y2);
        } else if (M_PI / 2 <= theta && theta < M_PI) {
            getClosest(m, b, flag, x_pos, y_pos, -25, 25, x2, y2);
        } else if (M_PI <= theta && theta < 3 * M_PI / 2) {
            getClosest(m, b, flag, x_pos, y_pos, -25, -25, x2, y2);
        } else {
            getClosest(m, b, flag, x_pos, y_pos, 25, -25, x2, y2);
        }
        double r = sqrt(pow(x_pos - x2, 2) + pow(y_pos - y2, 2));
        x.push_back(r * cos(theta));
        y.push_back(r * sin(theta));
    }

    std::map<std::string, std::string> kw;
    kw.insert(std::pair<std::string, std::string>("c", "r"));
    plt::scatter(x, y, 25.0, kw);
    plt::show();




    Eigen::MatrixXf p1 = Eigen::MatrixXf::Random(2, 5);

    // Simulate the generation of two clouds, motion-distorted:




    // (Add noise)

    // run the motion-distorted RANSAC to extract the desired transform, motion parameters?

    return 0;
}
