#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>
#include <iostream>
#include <association.hpp>

int main() {
    int N = 100;
    float theta = M_PI / 6;
    Eigen::Matrix2f R;
    R << cos(theta), sin(theta), -sin(theta), cos(theta);
    std::cout << R << std::endl;
    Eigen::Vector2f t = {1, 2};
    Eigen::MatrixXf p1 = Eigen::MatrixXf::Random(2, N);
    Eigen::MatrixXf p2 = R * p1;

    for (int i = 0; i < N; ++i) {
        p2.block(0, i, 2, 1) += t;
    }

    Eigen::MatrixXf Tf = Eigen::MatrixXf::Identity(3, 3);
    get_rigid_transform(p1, p2, Tf);
    std::cout << "T:" << std::endl << Tf << std::endl;

    return 0;
}
