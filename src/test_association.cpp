#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>
#include <association.hpp>

int main() {
    int N = 100;
    double theta = M_PI;
    Eigen::Matrix2d R;
    R << cos(theta), sin(theta), -sin(theta), cos(theta);
    std::cout << R << std::endl;
    Eigen::Vector2d t = {3, 5};
    Eigen::MatrixXd p1 = Eigen::MatrixXd::Random(2, N);
    Eigen::MatrixXd p2 = R * p1;

    for (int i = 0; i < N; ++i) {
        p2.block(0, i, 2, 1) += t;
    }

    Eigen::MatrixXd Tf = Eigen::MatrixXd::Identity(3, 3);
    get_rigid_transform(p1, p2, Tf);
    std::cout << "T:" << std::endl << Tf << std::endl;

    return 0;
}
